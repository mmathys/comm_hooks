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
    def __init__(self, model: nn.Module, device=None, c=60, r=5, k=60, momentum=1.0, sketchParamsLargerThan=0, sketchBiases=False):
        self.device = device
        
        for p in model.parameters():
            p.do_sketching = p.numel() >= sketchParamsLargerThan
           
        for m in model.modules():
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.do_sketching = sketchBiases

        grad_shape = 0
        sketch_shape = 0
        sketchMask = []
        for p in model.parameters():
            if p.requires_grad:
                size = torch.numel(p)
                grad_shape += size
                if p.do_sketching:
                    sketchMask.append(torch.ones(size))
                    sketch_shape += size
                else:
                    sketchMask.append(torch.zeros(size))

        sketchMask = torch.cat(sketchMask).bool().to(self.device) 
        assert sketchMask.numel() == grad_shape
        self.sketchMask = sketchMask
        self.grad_shape = grad_shape
        self.sketch_shape = sketch_shape

        self.u = torch.zeros(grad_shape, device=device)
        self.v = torch.zeros(grad_shape, device=device)
        self.momentum = momentum
        self.sketch = CSVec(d=sketch_shape, c=c, r=r, device=device)
        self.k = k
        self.r = r
        self.c = c


    def encode(self, gradient):
        self.u.mul_(self.momentum)
        self.u.add_(gradient)

        self.v.add_(self.u)

        v_masked = self.v[self.sketchMask]

        self.sketch.zero()
        self.sketch.accumulateVec(v_masked)
        table = self.sketch.table.clone()

        uncompressed = self.v[~self.sketchMask]
        assert uncompressed.size() == torch.Size([self.grad_shape - self.sketch_shape])

        return torch.cat([table.view(-1), uncompressed])

    def decode(self, payload):
        table_len = self.r * self.c
        sketch_table = payload[:table_len].view(self.r, self.c)
        uncompressed = payload[table_len:]

        # deal with compressed gradients
        self.sketch.zero()
        self.sketch.table = sketch_table

        gradient = torch.zeros(self.grad_shape)
        unsketched = self.sketch.unSketch(k=self.k)
        gradient[self.sketchMask] = unsketched

        self.u[gradient.nonzero()] = 0
        self.v[gradient.nonzero()] = 0

        # deal with non-compressed gradients (bias)
        assert uncompressed.size() == torch.Size([self.grad_shape - self.sketch_shape])
        gradient[~self.sketchMask] = uncompressed
        self.v[~self.sketchMask] = 0

        return gradient

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def example(rank, world_size, device="cpu"):
    setup(rank, world_size)
    print(f"init rank {rank}")
    # create local model
    model = torch.nn.Sequential(torch.nn.Linear(200, 1))
    # construct DDP model
    ddp_model = DDP(model)

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
    state = SketchState(model, device=device)
    ddp_model.register_comm_hook(state=state, hook=encode_and_decode)

    # prepare sample dataset
    X = torch.randn(1000, 200)
    #y = torch.randn(20, 10)
    y = torch.zeros(1000, 1)
    
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    # important: do not use momentum in the optimizer, already handled.
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.0001)

    for i in range(100):
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
    main()