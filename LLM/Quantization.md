# Quantization
Quantization은 실수형 변수(floating-point type)을 정수형 변수(integer or fixed point)로 변환하는 과정을 뜻한다.

![](https://cdn.discordapp.com/attachments/874897301292875836/1135196434816303224/2023-07-30_10.05.15.png)

위 그림과 같이 Quantization을 적용하면 일반적으로 많이 사용하는 FP32 타입의 파라미터를 INT8 형태로 변환한 다음에 실제 inference를 하게 된다.

이 작업은 weights나 activation function의 값이 어느 정도의 범위 안에 있다는 것을 가정하여 이루어 지는 모델 경량화 방법이다. 

![](https://cdn.discordapp.com/attachments/874897301292875836/1135197231671156756/2023-07-30_10.08.24.png)

위 그림과 같이 floating point로 학습한 모델의 weight 값이 -10 ~ 30 범위에 있다고 가정한다. 이 때, 최소값인 -10을 unit8의 0에 대응시키고 30을 unit8의 최대값인 255에 대응시켜서 사용한다면 32bit 자료형이 8bit자료형으로 줄어들기 때문에 전체 메모리 사용량 및 수행 속도가 감소하는 효과를 얻을 수 있다.

### Quantization 장점
* 이와 같이 Quantization을 통해 효과적인 모델 최적화를 할 수 잇는데, float타입을 int형으로 줄이면서 용량을 줄일 수 있고 bit수를 줄임으로써 계산 복잡도도 줄일 수 있다.(일반적으로 정수형 변수의 bit수를 N배 줄이면 곱셈 복잡도는 N*N배로 줄어든다.)

* 또한 정수형이 하드웨어에 더 친화적이기 때문에 Quantization을 통한 최적화가 필요하다.

* 정리하면 **모델 사이즈 축소**, **모델의 연산량 감소**, **효율적인 하드웨어 사용**이 Quantization의 주요 목적이라고 말할 수 있다.

### 딥러닝 모델에서 Quantization의 역할

![](https://cdn.discordapp.com/attachments/874897301292875836/1135199861688512602/2023-07-30_10.18.52.png)

ResNet-34를 2bit로 표현했을 때의 Top-1 Accuracy가 ResNet-18을 4-bit로 표현하였을 때보다 성능이 좋다. 이 때 모델 사이즈는 ResNet-34가 더 작은 것도 확인할 수 있다.

즉 작은 네트워크로 quantization을 대충하는 것보다 큰 네트워크로 quantization을 더 잘하는 게 성능 및 모델 사이즈 측면에서 더 좋을 수 있다.

FP32->INT8 로 변환 시 보통 model size는 1/4가 되고, inference speed 는 2~4배가 빨라지며 memory bandwidth도 2~4배 가벼워진다.

### Pytorch Quantization 추가 예정