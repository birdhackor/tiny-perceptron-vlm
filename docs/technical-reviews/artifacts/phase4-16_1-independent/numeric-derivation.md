16.1 independent arithmetic, CPU/inference scope

ModelConfig(width=8): vocabulary=264, max_length=128, layers=1, heads=1, untied output; default dtype FP32.
Embedding: 264*8 = 2112.
Position table: 128*8 = 1024.
Attention Q,K,V,out, each bias=False: 4*(8*8) = 256.
Three LayerNorms (norm1, norm2, final_norm), each gamma+beta: 3*(2*8) = 48.
DenseFFN hidden width 4*8=32: up (8*32+32)=288, down (32*8+8)=264; subtotal 552.
Output untied bias=False: 8*264 = 2112.
Sum 2112+1024+256+48+552+2112=6104 elements; FP32 element_size=4 bytes; 6104*4=24416 bytes.
This counts parameter element payloads, not allocator, activations, cache, gradients or optimizer states.
Logits axes: batch=1, positions=3 or 12, vocabulary=264, so (1,3,264) or (1,12,264).
Nine sorted samples: middle is rank (9+1)/2=5. Four sorted values [1,2,3,20]: median=(2+3)/2=2.5; arithmetic mean=(1+2+3+20)/4=6.5.
1 MiB = 2^20 bytes = 1,048,576 bytes (GNU Coreutils official Block-size M/MiB paragraph independently read).
All integer counts/exact byte arithmetic require exact equality. Softmax row sum CPU assertion uses torch.allclose default rtol=1e-5, atol=1e-8; that auxiliary check does not claim an attention numerical-accuracy bound.
