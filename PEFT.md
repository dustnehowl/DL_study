# Parameter Efficient Fine-Tuning(PEFT)

PEFT는 모델의 모든 파라미터를 튜닝하는 것이 아닌 일부 파라미터를 튜닝함으로써 모델의 성능을 적은 자원으로도 높게 유지하는 방법론이다.

PEFT 기법중에서 가장 많이 알려진 것은 LoRA 라는 기법이다. LoRA와 IA3라는 방법론에 대해 알아보자.

## LLM의 발전과 PEFT의 필요성
최근 LLM이 등장함에 따라 다양한 문제를 쉽게 언어모델에 해결할 수 있게 되었다.

풀고자 하는 문제에 대한 몇가지 예시만 few-shot으로 모델에 미리 입력을 해주면 in-context learning(ICL)에 의해 모델을 따로 튜닝할 필요 없이 쉽게 문제를 풀 수 있다.

하지만 이런 ICL 은 매번 미리 예제를 입력해주어야하기 때문에 계산비용, 메모리비용, 저장비용등이 발생하게 된다는 단점이 있다.

또한 어떤 연구에서는 incorrect labels를 예제로 넣어주더라도 문제를 잘 해결하기도 하더라 라는 내용이 있어 온전히 신뢰하기 어렵다.

PEFT는 이러한 단점을 보완하는 대안적인 패러다임 중 하나이고 적은양 파라미터를 학습함으로써 빠른 시간 내에 새로운 문제를 거의 비슷한 성능으로 풀 수 있게 하자는데에 목적이 있다.

언어 모델처럼 매우 많은 수의 파라미터를 쓰는 모델이 사실은 적은 수의 파라미터를 튜닝해도 비슷한 성능을 낼 수 있다는 선행연구들이 있었고, 이러한 연구를 기반으로 현재 PEFT를 위한 다양한 방법론들이 연구되고 있다.

방법론 중 가장 유명한 것중 하나가 LoRA고, 최근에는 LoRA를 개선한 방법론들도 많이 나오고 있는 상황이다.

LoRA는 huggingface 라이브러리에 구현되어 있고, IA3은 NVIDIA NeMo에 구현되어 있다.

## PEFT 기법들
초기에 PEFT를 위해 제안되었던 방법은 어댑터(adapters)를 사용하는 것이다. adapters란 기존에 이미 학습이 완료된 모델 사이사이에 **학습 가능한 작은 feed-forward networks**를 삽입하는 구조를 말한다.

이때 pre-trained model의 weights는 고정해놓고 학습 가능한 네트워크만 아키텍쳐 중간중간마다 추가함으로서 적은 수의 파라미터로 모델을 튜닝하는 것이다.

현재는 Microsoft에서 공개한 LoRA라는 방법론이 현재로서는 제일 유명하다.

## LoRA
Low-Rank Adaptation(LoRA) 의 개념을 간단하게 설명하면, 고정된 weights를 갖는 pretrained model에 학습이 가능한 rank decomposition 행렬을 삽입한 것으로 중간중간 학습이 가능한 파라미터를 삽입했다는 점에서 adapters와 비슷하지만 구조적으로 조금 다르다.

적은 양의 파라미터로 모델을 튜닝하는 방법론이기 때문에 적은 수의 GPU로 빠르게 튜닝할 수 있다는 장점이 있다. LoRA에서 나온 rank decomposition이라는 말은 아래의 그림처럼 행렬의 차원을 r만큼 줄이는 행렬과 다시 원래 크기로 키워주는 행렬의 곱으로 나타내는 것을 의미한다.


![LoRA](https://cdn.discordapp.com/attachments/874897301292875836/1134084548465020958/2023-07-27_8.27.01.png)

그림처럼 레이어 사이에 존재하는 hidden states h 에 값을 더해줄 수 있는 파라미터를 추가해줘서 모델의 출력 값을 원하는 타겟 레이블에 맞게 튜닝하는 것이 LoRA의 핵심 개념이다.

코드상으로는 아래와 같이 구현할 수 있다. 기존의 모델에서 사용하던 Linear Layer를 LoRA의 로직이 적용된 커스텀 클래스로 교체하여 구현한다.

if self.r > 0: 라는 부분이 LoRA가 적용된 부분이다.
```python
class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
```
위 코드에서 주목할 부분은 eval 함수이다.
```python
def eval(self):
    def T(w):
        return w.T if self.fan_in_fan_out else w
    nn.Linear.eval(self)
    if self.merge_weights and not self.merged:
        # Merge the weights and mark it
        if self.r > 0:
            self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling # 행렬을 합치는 부분
        self.merged = True
```
LoRA가 행렬 연산을 기반으로 하기 때문에 기존 행렬 W_0를 LoRA에서 사용하는 A, B 행렬을 기반으로 다음과 같이 재구성할 수 있다.

### W = W_0 + BA

이렇게 함으로써 얻을 수 있는 이점은 새롭게 학습한 파라미터를 기존에 학습된 pretrained model에 합쳐줌으로서 추가적인 연산이 피룡하지 않게 되어 속도도 그대로 유지하면서 아키텍쳐의 변경도 필요없어지게 된다.

## Alpaca-LoRA
최근에 유행하는 LLaMA의 변형인 Alpaca에도 LoRA가 적용된 오픈소스 프로젝트들이 공개되고 있다.

huggingface에서 공개한 PEFT를 이용하면 아래와 같이 간단하게 적용해볼 수 있다.
```python
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

def train(...):
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
```

## IA3
Infused Adapter by Inhibiting ans Amplifying Inner Activations)의 개념을 간단하게 설명하면 LoRA와 비슷한 방법으로 적은 파라미터만을 추가하여 모델을 튜닝할 수 있는 방법론이다.

이름에 나와있듯이 NN의 Inner Activation을 줄이기도하고 늘리기도 하는 어댑터를 중간에 삽입하는 방법론이다.

LoRA의 경우 hidden state에 새로운 값을 더해주는 기법이라면, IA3의 경우에는 Self-Attention, Cross-Attention에서의 Key, Value 값을 rescale해주는 벡터와 position-wise feed-forward network에 값을 rescale을 해주는 벡터를 추가해 모델을 튜닝해주는 기법이다.
![](https://cdn.discordapp.com/attachments/874897301292875836/1134089265714839603/2023-07-27_8.45.44.png)

IA3는 기존에 공개된 LoRA보다 적은 파라미터를 사용하면서 높은 성능을 내는 것으로 알려져 있으며, GPT-3를 in-context learning했을 때보다도 성능이 좋다라고 주장하고 있다. 학습시간도 A100하나로 30분만에 튜닝할 수 있다고 한다.

IA3도 LoRA와 마찬가지로 Linear Layer를 커스텀 구현체로 변경함으로서 구현이 가능하며, LoRA Layer에 대한 configuration을 수정하여 구현할 수 있다. 

## 결론
최근 몇년간 Scaling laws에 따라 언어모델의 크기가 점점 커지고 있는 경향이다.

이러한 상황에서 인프라가 없는 개인이나 기업은 적은 자원으로도 모델을 빠르게 튜닝할 수 있는 방법이 필요하다.

PEFT는 그러한 목적을 이룰 수 있는 훌륭한 대안이 될 수 있다. 