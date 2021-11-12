`cpu: Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz` + `GTX1080`

```
goos: linux
goarch: amd64
pkg: github.com/xaionaro/cuFFT-gRPC/client_example/go
cpu: Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize2/isInversefalse-12         	    2677	    427550 ns/op	    5176 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize2/isInversetrue-12          	    3044	    446440 ns/op	    5174 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize10/isInversefalse-12        	    2808	    415652 ns/op	    5685 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize10/isInversetrue-12         	    3102	    418303 ns/op	    5687 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize100/isInversefalse-12       	    3090	    454825 ns/op	   12199 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize100/isInversetrue-12        	    3006	    435753 ns/op	   12198 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize1000/isInversefalse-12      	    2516	    484651 ns/op	   71370 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize1000/isInversetrue-12       	    2144	    481135 ns/op	   71493 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize10000/isInversefalse-12     	    1309	    909255 ns/op	  680445 B/op	     114 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize10000/isInversetrue-12      	    1419	    914604 ns/op	  679473 B/op	     114 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize100000/isInversefalse-12    	     243	   4605540 ns/op	 6525320 B/op	     159 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize100000/isInversetrue-12     	     277	   4490573 ns/op	 6528000 B/op	     152 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize2/isInversefalse-12        	    2758	    420431 ns/op	    6359 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize2/isInversetrue-12         	    3025	    420275 ns/op	    6360 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize10/isInversefalse-12       	    2960	    419066 ns/op	   12202 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize10/isInversetrue-12        	    2620	    425644 ns/op	   12197 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize100/isInversefalse-12      	    2559	    465245 ns/op	   71398 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize100/isInversetrue-12       	    2827	    477001 ns/op	   71368 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize1000/isInversefalse-12     	    1497	    862510 ns/op	  680063 B/op	     110 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize1000/isInversetrue-12      	    1506	    855254 ns/op	  680521 B/op	     110 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize10000/isInversefalse-12    	     276	   4433139 ns/op	 6523441 B/op	     146 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize10000/isInversetrue-12     	     255	   4428196 ns/op	 6532159 B/op	     147 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize2/isInversefalse-12       	    3028	    416297 ns/op	   18379 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize2/isInversetrue-12        	    2923	    423145 ns/op	   18380 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize10/isInversefalse-12      	    2774	    463804 ns/op	   71403 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize10/isInversetrue-12       	    2607	    482183 ns/op	   71468 B/op	     100 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize100/isInversefalse-12     	    1340	    884119 ns/op	  679590 B/op	     110 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize100/isInversetrue-12      	    1452	    869126 ns/op	  679868 B/op	     110 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize1000/isInversefalse-12    	     273	   4390327 ns/op	 6523142 B/op	     145 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize1000/isInversetrue-12     	     279	   4385040 ns/op	 6516561 B/op	     145 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize2/isInversefalse-12             	  263046	      3826 ns/op	     112 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize2/isInversetrue-12              	  360045	      3734 ns/op	     112 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize10/isInversefalse-12            	  346142	      3765 ns/op	     368 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize10/isInversetrue-12             	  300634	      3801 ns/op	     368 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize100/isInversefalse-12           	   39734	     33433 ns/op	    3632 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize100/isInversetrue-12            	   40736	     34063 ns/op	    3632 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize1000/isInversefalse-12          	   10048	    126661 ns/op	   32816 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize1000/isInversetrue-12           	   10747	    120681 ns/op	   32816 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize10000/isInversefalse-12         	    1388	    985044 ns/op	  327728 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize10000/isInversetrue-12          	    6264	    205484 ns/op	  327728 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize100000/isInversefalse-12        	       1	1892998095 ns/op	 3211312 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize100000/isInversetrue-12         	       1	1792307944 ns/op	 3211312 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize2/isInversefalse-12            	  285892	      4915 ns/op	     184 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize2/isInversetrue-12             	  280501	      4700 ns/op	     184 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize10/isInversefalse-12           	  279920	      4951 ns/op	     440 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize10/isInversetrue-12            	  270286	      5004 ns/op	     440 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize100/isInversefalse-12          	   33462	     38451 ns/op	    3704 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize100/isInversetrue-12           	   28857	     38743 ns/op	    3704 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize1000/isInversefalse-12         	    8096	    165394 ns/op	   32888 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize1000/isInversetrue-12          	    6421	    169354 ns/op	   32888 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize10000/isInversefalse-12        	     736	   1491939 ns/op	  327800 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize10000/isInversetrue-12         	    1837	    732922 ns/op	  327800 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize2/isInversefalse-12           	  102129	     12219 ns/op	     904 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize2/isInversetrue-12            	  100525	     12484 ns/op	     904 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize10/isInversefalse-12          	  100338	     13482 ns/op	    1160 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize10/isInversetrue-12           	   97245	     13904 ns/op	    1160 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize100/isInversefalse-12         	   15321	     70989 ns/op	    4424 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize100/isInversetrue-12          	   19006	     71688 ns/op	    4424 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize1000/isInversefalse-12        	    2248	    541177 ns/op	   33608 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize1000/isInversetrue-12         	    2467	    525441 ns/op	   33608 B/op	     104 allocs/op
PASS
ok  	github.com/xaionaro/cuFFT-gRPC/client_example/go	92.730s
?   	github.com/xaionaro/cuFFT-gRPC/client_example/go/protobufgen/github.com/xaionaro/cuFFT-gRPC/protobuf	[no test files]
```