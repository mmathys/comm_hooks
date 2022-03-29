import os
from csvec import CSVec
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# Implements SketchSGD encoding and decoding. This is can be used for the stateful
# hook.
class SketchState:
    def __init__(self, model: nn.Module, device=None, c=10, r=10, k=50, momentum=1.0):
        grad_shape = 0
        for p in model.parameters():
            if p.requires_grad:
                grad_shape += torch.numel(p)
        self.device = device
        self.u = torch.zeros(grad_shape, device=device)
        self.v = torch.zeros(grad_shape, device=device)
        self.momentum = momentum
        self.sketch = CSVec(d=grad_shape, c=c, r=r)
        self.k = k

    def encode(self, gradient):
        self.u.mul_(self.momentum)
        self.u.add_(gradient)

        self.v.add_(self.u)

        self.sketch.zero()
        self.sketch.accumulateVec(self.v)
        return self.sketch.table.clone() 

    def decode(self, sketch_table):
        self.sketch.zero()
        self.sketch.table = sketch_table
        gradient = self.sketch.unSketch(k=self.k)
        self.u[gradient.nonzero()] = 0
        self.v[gradient.nonzero()] = 0

        return gradient

def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])

    # Define a encode-decode hook for the gradient bucket.
    # The GradientBucket.buffer returns the gradient vector of the whole network.
    # this hook takes any object as state. in our case: our SketchState, which holds
    # the variables for error accumulation and more.
    def encode_and_decode(state: SketchState, bucket: dist.GradBucket):
        # encode full gradient into a count sketch.
        encoded_tensor = state.encode(bucket.buffer())

        # All-reduce all count sketches using sums.
        fut = dist.all_reduce(encoded_tensor, group=dist.group.WORLD, async_op=True, op=ReduceOp.SUM).get_future()
        
        # Define the then callback to decode.
        def decode_fut(fut):
            # decode count summed count sketches into a gradient.
            decoded_tensor = state.decode(fut.value()[0])
            return decoded_tensor
        
        return fut.then(decode_fut)
    
    # initialize the state 
    state = SketchState(model, device=rank)
    ddp_model.register_comm_hook(state=state, hook=encode_and_decode)

    # prepare sample dataset
    X = torch.randn(20, 10).to(rank)
    y = torch.randn(20, 10).to(rank)

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    # important: do not use momentum in the optimizer, already handled.
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for i in range(20):
        optimizer.zero_grad()

        # forward pass
        y_pred = ddp_model(X)
        
        # calculate loss
        loss = loss_fn(y, y_pred)
        print("[{}] Loss: {}".format(i, loss.item()))

        # backward pass
        loss.backward()
        
        # update parameters
        optimizer.step()

def main():
    world_size = 1
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be set
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()