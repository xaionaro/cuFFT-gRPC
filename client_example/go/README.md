`cpu: Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz` + `GTX1080`

```
goos: linux
goarch: amd64
pkg: github.com/xaionaro/cuFFT-gRPC/client_example/go
cpu: Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize2/isInversefalse-12         	    2024	    502320 ns/op	    5498 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize2/isInversetrue-12          	    2635	    507605 ns/op	    5496 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize10/isInversefalse-12        	    2481	    506865 ns/op	    5498 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize10/isInversetrue-12         	    2568	    529910 ns/op	    5498 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize100/isInversefalse-12       	    2458	    527104 ns/op	    5496 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize100/isInversetrue-12        	    2056	    534049 ns/op	    5496 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize1000/isInversefalse-12      	    2192	    557449 ns/op	    5494 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize1000/isInversetrue-12       	    2268	    535245 ns/op	    5494 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize10000/isInversefalse-12     	    1542	    759191 ns/op	    5496 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize10000/isInversetrue-12      	    1603	    761219 ns/op	    5494 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize100000/isInversefalse-12    	     434	   2751149 ns/op	    5501 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1/taskSize100000/isInversetrue-12     	     438	   2757025 ns/op	    5501 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize2/isInversefalse-12        	    2433	    514198 ns/op	    5495 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize2/isInversetrue-12         	    2379	    509493 ns/op	    5494 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize10/isInversefalse-12       	    2172	    522961 ns/op	    5495 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize10/isInversetrue-12        	    2193	    515961 ns/op	    5494 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize100/isInversefalse-12      	    2161	    526882 ns/op	    5494 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize100/isInversetrue-12       	    2233	    537532 ns/op	    5495 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize1000/isInversefalse-12     	    1842	    713586 ns/op	    5495 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize1000/isInversetrue-12      	    1696	    727075 ns/op	    5498 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize10000/isInversefalse-12    	     466	   2531895 ns/op	    5507 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10/taskSize10000/isInversetrue-12     	     438	   2598281 ns/op	    5501 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize2/isInversefalse-12       	    2246	    501010 ns/op	    5494 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize2/isInversetrue-12        	    2527	    514395 ns/op	    5495 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize10/isInversefalse-12      	    2188	    526576 ns/op	    5495 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize10/isInversetrue-12       	    2433	    528766 ns/op	    5494 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize100/isInversefalse-12     	    1628	    733265 ns/op	    5495 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize100/isInversetrue-12      	    1570	    726051 ns/op	    5494 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize1000/isInversefalse-12    	     445	   2578016 ns/op	    5501 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize100/taskSize1000/isInversetrue-12     	     448	   2582529 ns/op	    5501 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1000/taskSize2/isInversefalse-12      	    2229	    559007 ns/op	    5493 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1000/taskSize2/isInversetrue-12       	    2185	    555992 ns/op	    5493 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1000/taskSize10/isInversefalse-12     	    1648	    714818 ns/op	    5497 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1000/taskSize10/isInversetrue-12      	    1689	    727918 ns/op	    5496 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1000/taskSize100/isInversefalse-12    	     441	   2636811 ns/op	    5504 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize1000/taskSize100/isInversetrue-12     	     460	   2527488 ns/op	    5500 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10000/taskSize2/isInversefalse-12     	    1317	    937002 ns/op	    5496 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10000/taskSize2/isInversetrue-12      	    1203	    941741 ns/op	    5496 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10000/taskSize10/isInversefalse-12    	     446	   2558074 ns/op	    5500 B/op	     104 allocs/op
BenchmarkCuFFTThroughGRPC/batchSize10000/taskSize10/isInversetrue-12     	     450	   2668320 ns/op	    5501 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize2/isInversefalse-12                      	  327370	      3965 ns/op	     112 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize2/isInversetrue-12                       	  283953	      3822 ns/op	     112 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize10/isInversefalse-12                     	  313214	      4044 ns/op	     368 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize10/isInversetrue-12                      	  297220	      4341 ns/op	     368 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize100/isInversefalse-12                    	   58656	     20145 ns/op	    3632 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize100/isInversetrue-12                     	   37874	     32241 ns/op	    3632 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize1000/isInversefalse-12                   	    9399	    139763 ns/op	   32816 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize1000/isInversetrue-12                    	    8181	    134288 ns/op	   32816 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize10000/isInversefalse-12                  	    3296	    366257 ns/op	  327728 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize10000/isInversetrue-12                   	    6000	    226026 ns/op	  327728 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize100000/isInversefalse-12                 	       1	2003078897 ns/op	 3211312 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize1/taskSize100000/isInversetrue-12                  	       1	1918439908 ns/op	 3211312 B/op	       5 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize2/isInversefalse-12                     	  254815	      5153 ns/op	     184 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize2/isInversetrue-12                      	  253800	      5049 ns/op	     184 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize10/isInversefalse-12                    	  227989	      5118 ns/op	     440 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize10/isInversetrue-12                     	  221349	      5425 ns/op	     440 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize100/isInversefalse-12                   	   49286	     25027 ns/op	    3704 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize100/isInversetrue-12                    	   31707	     37794 ns/op	    3704 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize1000/isInversefalse-12                  	    7192	    178296 ns/op	   32888 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize1000/isInversetrue-12                   	    6224	    179412 ns/op	   32888 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize10000/isInversefalse-12                 	    1293	    908010 ns/op	  327800 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize10/taskSize10000/isInversetrue-12                  	    1608	    775322 ns/op	  327801 B/op	      14 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize2/isInversefalse-12                    	   90255	     13219 ns/op	     904 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize2/isInversetrue-12                     	   95956	     13580 ns/op	     904 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize10/isInversefalse-12                   	   75136	     14693 ns/op	    1160 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize10/isInversetrue-12                    	   80050	     15250 ns/op	    1160 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize100/isInversefalse-12                  	   17337	     67282 ns/op	    4424 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize100/isInversetrue-12                   	   15604	     77032 ns/op	    4424 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize1000/isInversefalse-12                 	    2188	    563397 ns/op	   33608 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize100/taskSize1000/isInversetrue-12                  	    2169	    558505 ns/op	   33608 B/op	     104 allocs/op
BenchmarkFFTWDirectly/batchSize1000/taskSize2/isInversefalse-12                   	   12297	     99158 ns/op	    8104 B/op	    1004 allocs/op
BenchmarkFFTWDirectly/batchSize1000/taskSize2/isInversetrue-12                    	   12253	     96384 ns/op	    8104 B/op	    1004 allocs/op
BenchmarkFFTWDirectly/batchSize1000/taskSize10/isInversefalse-12                  	   10000	    116187 ns/op	    8360 B/op	    1004 allocs/op
BenchmarkFFTWDirectly/batchSize1000/taskSize10/isInversetrue-12                   	   10000	    108434 ns/op	    8360 B/op	    1004 allocs/op
BenchmarkFFTWDirectly/batchSize1000/taskSize100/isInversefalse-12                 	    2288	    517462 ns/op	   11624 B/op	    1004 allocs/op
BenchmarkFFTWDirectly/batchSize1000/taskSize100/isInversetrue-12                  	    2529	    470873 ns/op	   11624 B/op	    1004 allocs/op
BenchmarkFFTWDirectly/batchSize10000/taskSize2/isInversefalse-12                  	    1429	    939113 ns/op	   80104 B/op	   10004 allocs/op
BenchmarkFFTWDirectly/batchSize10000/taskSize2/isInversetrue-12                   	    1159	    947475 ns/op	   80104 B/op	   10004 allocs/op
BenchmarkFFTWDirectly/batchSize10000/taskSize10/isInversefalse-12                 	    1204	   1097685 ns/op	   80360 B/op	   10004 allocs/op
BenchmarkFFTWDirectly/batchSize10000/taskSize10/isInversetrue-12                  	    1209	   1041937 ns/op	   80360 B/op	   10004 allocs/op
PASS
ok  	github.com/xaionaro/cuFFT-gRPC/client_example/go	112.905s
?   	github.com/xaionaro/cuFFT-gRPC/client_example/go/protobufgen/github.com/xaionaro/cuFFT-gRPC/protobuf	[no test files]
```
