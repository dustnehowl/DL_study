# In-Context Learning

in-context learning은 주어지는 context, 예시에 따라 zero-shot, one-shot, few-shot으로 나눌 수 있다.

## zero-shot learning
zero-shot 은 예시가 없거나 task가 주어졌을 때 수행가능한 경우를 의미한다. GPT와 같은 언어모델들이 unsupervised learning을 수행하나, GPT-2이후부터 task에 대한 학습이 이루어졌기 때문에 대부분의 nlp task들은 별도의 in-context learning 없이 zero-shot으로 수행가능하다. 번역이나, 요약, 분류등의 task등이 zero-shot으로 수행가능한 부분이다.

```python
# zero-shot example
Prompt: 빨간 사과가 영어로 뭐야?
GPT: "Red Apple"
```

## one-shot learning
one-shot 은 하나의 예시를 들어주고 task를 수행하는 경우를 말한다.
```python
# one-shot example
Prompt: 빨간 사과는 red 사과라고 할게. 노란 바나나는?
GPT: 노란 바나나는 "yellow 바나나"입니다.
```

## Few-shot learning
Few-shot learning은 예시를 여럿 주고 task를 수행하는 경우를 말한다.
```python
# few-shot example
Prompt: 빨간 사과는 red 사과라고 할께,
노란 바나나는 yellow 바나나야,
그럼 노란 사과는?

GPT: 노란 사과는 "yellow 사과"입니다.
```

one-shot과 few-shot의 차이는 예시를 하나 또는 여럿 들어주는 것의 차이입니다. 예시가 적어도 잘 알아듣는다는 것은 이미 그 모델이 해당 상황에 대한 인지가 있다고 유추해볼 수 있으며, 예시가 많았을 때 잘할 수 있다는 것은 문맥적 이해 능력이 좋다고 볼 수 있다.

이러한 zero-shot, one-shot, few-shot에 대한 평가는 GPT-3논문에서 다양하게 이루어졌다. 일부 task에 대해서는 fine tuning을 넘어서는 경우도 있고 그렇지 못한 경우들도 있으나, 잘 작동할 수 있다는 점을 많은 예시와 테스트를 통해 증명했다.

## Prompt Engineering

GPT-3 처럼 거대한 LLM이 나타나면서 ICL이 더 중요해지고 있다. LLM은 입력으로 준 텍스트에 이어서 단어들을 예측하는 알고리즘이기 때문에 LLM에게 어떻게 입력을 주는지에 따라 LLM의 답의 품질도 달라진다.

많은 사람들이 좋은 Prompt를 작성하는 법을 연구하기 시작했다. 좋은 Prompt란 LLM이 기대한대로 좋은 답을 주는 Prompt이다. 이런 Prompt를 찾는 노력을 Prompt Engineering이라고 하고, 실제로 많은 회사에서 몸값이 오르고 있다.