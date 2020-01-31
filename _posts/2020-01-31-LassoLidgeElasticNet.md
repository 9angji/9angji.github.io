# Lasso, Lidge, ElasticNet

**정규화(Regularization)**

- 회귀계수가 가질 수 있는 값에 제약조건을 부여
- 정규화항을 통해 모델에 미치는 차원의 수 감소→ Overfitting 방지
- 오차의 기대값→ Bias, Variance로 분해 가능
- 정규화는 variance를 감소시켜 일반화 성능을 높이는 방법

<img src="/data/img/2020-01-31/ra_ri_el (1).png">

- 좌측의 그림은 학습데이터를 잘 맞추고 있지만, 미래 데이터가 조금만 바뀌어도 예측값이 들쭉날쭉할 수 있음.
- 반면, 우측 그림은 가장 강한 수준의 정규화를 수행한 결과로, 학습데이터에 대한 설명력을 다소 포기하는 대신 미래 데이터 변화에 상대적으로 안정적인 결과를 냄

---

**Bias-Variance Decomposition**

- 정규화(Regularization), 앙상블 기법(ensemble)의 이론적 배경
- 학습에 쓰지 않은 미래데이터에 대한 오차의 기대값을 모델의 Bias와 Variance로 분해가능 하다는 내용

F*(기대값: 0)가 찾아야할 궁극적 모델일 때, 이를 확률모형으로 나타내면 아래와 같다.

$$y=f^{\ast }\left( x\right) +\varepsilon,\quad \varepsilon \sim N\left( \sigma^{2}\right) \\ \\ \sigma^{2} : 이론적인 오차,\,natural error$$

궁극의 모델 F*를 항상 찾아낼 수 있는 것은 아니다. 

학습데이터에서 찾아낸 모델을 F-hat이라 할 때, F-hat은 모델을 구축할 때 마다 달라질 수 있다.

<img src="/data/img/2020-01-31/ra_ri_el (2).png">

여러개의 F-hat모델로 부터 F-bar(F-hat 모델의 평균)모델을 도출할 수 있음.

$$\overline {F}\left( x\right) =E\left[ \widehat {F}_{i}\left( x\right) \right] $$

학습데이터로부터 F-hat을 만들고, 이를 바탕으로 미래데이터 $x_0$를 예측해야 한다고 가정할 때, 이 데이터에 대한 오차의 기대값을 수식으로 나타내면 아래와 같다. 

$$ExpectedMSE=E[( y-\hat {F}( x)) ^{2}|x=x_{0}] \\ =E[\{F^*(x_0)+\varepsilon-\hat{F}(x_0)\}^2] \\ =E[\{F^*(x_0)-\hat{F}(x_0)\}^2]-+2\{F^*(x_0)-\hat{F}(x_0)\}E[\varepsilon]+E[\varepsilon{}^2]$$

$E(\varepsilon)=0$ 이므로 $Var(\varepsilon)=E(\varepsilon{^2})$ 이다. 

따라서, 아래와 같이 변형할 수 있다. 

$$Expected MSE= E[\{F^*(x_0)-\hat{F}(x_0)\}^2]+\sigma{^2} \\ =E[\{F^*(x_0)-\bar{F}(x_0)+\bar{F}(x_0)-\hat{x_0}\}^2]+\sigma{^2}$$

 이 때, 

$$E[\bar{F}(x_0)-\hat{x_0}]=\bar{x_0}-\bar{F}(x_0)=0$$

이기 때문에, 원래의 식에 정리 대입하면 다음과 같다. 

$$Expected MSE = E[\{F^*(x_0)-\bar{F}(x_0)\}^2]+E[\{\bar{F}(x_0)-\hat{F}(x_0)\}^2]+\sigma{^2} \\ =\{F^*(x_0)-\bar{F}(x_0)\}^2+E[\bar{F}(x_0)-\hat{F}(x_0)\}^2]+\sigma{^2} \\ =Bias^2(\hat{F}(x_0))+Var(\hat{F}(x_0))+\sigma{^2}$$

위 식의 의미: 

- Bias는 여러 모델의 예측 평군(F-bar)과 실제 정답(y)의 편차.
- Var는 개별모델의 예측값(F-hat)과 여러 모델의 예측 평균(F-bar)과의 편차제곱의 기대값
- 따라서, 임의의 미래데이터 $x_0$의 기대값은 모델의 Bias, Variance, Natural Error 세 요소를 분해할 수 있음.

직관적으로 살펴보면: 

<img src="/data/img/2020-01-31/ra_ri_el (3).png">

- 파란 원의 중심은 True Function $F^*$를 의미
- 파란 원은 실제값 y가 가질 수 있는 범위
- 노란 워의 중심(빨간 점)은 F-hat의 평균(F-bar)
- 빨간 실선은 F-hat이 예측하는 값의 범위
- 즉, Bias 와 Variance가 작을 수록 True Function에 가까워짐

<img src="/data/img/2020-01-31/ra_ri_el (4).png">

- 첫번째 그림 : 예측값(파란색)의 평균이 과녁(Truth)와 멀리 떨어져 있어 Variance가 큼
- 네번째 그림 : Bias, Variance 모두 작기 때문에 가장 이상적
- 두번째 그림 : 예측값 평균이 과녁과 멀지 않아 Bias는 작음

    → 뉴럴 네트워크, SVM, K-NN(small k) 등 튜닝만 잘하면 예측률이 높아질 수 있는 모델

    → 부스팅, 라쏘회귀 등의 정규화 기법이 variance를 줄여 성능 향상을 추구하는 기법 

- 세번째 그림 : 예측값들이 모여 있어 Variance는 작으나 Bias가 큼

    → 로지스틱 회귀, LDA, K-NN(large k) 등 데이터 노이즈에 비교적 강건한 모델

    ---

- 일반적인 회귀 방법에서 비용함수는 MSE를 최소화하는 방향으로 회귀계수를 추정함. 일반적인 회귀방법에서는 데이터의 Feature수가 많을 수록 overfitting에 대한 위험성이 커짐
- 이를 막기 위해, 정규화 항 사용
    - MSE+(regular term)
    - 비용함수는 MSE를 포함해 regular term또한 최소화하기 위해 가중치가 낮은 항은 정규화 방법에 따라 0으로 수렴하도록 하거나 0에 가까운 수가 되어 모델에 미치는 영향이 덜해지게됨
- 대표적인 정규화 회귀 방법에 3가지가 있음
    - 릿지회귀(Lidge)
    - 라쏘회귀(Lasso)
    - 엘라스틱넷(elasticNet)

**라쏘회귀(Lasso)**

- 라쏘 회귀는 L1-Norm을 사용한 회귀
- 특성값의 계수가 매우 낮으면 0으로 수렴하도록 하여 특성을 제거함

$$J\left( \theta \right) =MSE\left( \theta \right) +\alpha \sum _{i=1}^{n}\left| \theta _{i}\right| $$

**릿지회귀(Lidge)**

- 릿지 회귀는 L2-Norm을 사용한 회귀
- 영향을 거의 미치지 않는 특성에 대하여 0에 가까운 가중치를 부여

$$J\left( \theta \right) =MSE\left( \theta \right) +\alpha \frac {1} {2}\sum _{i=1}^{n}\theta _{i}^{2}$$

**엘라스틱넷(elasticNet)**

- 라쏘회귀와 릿지회귀의 최적화 지점이 서로 다르기 때문에 두 정규화항을 합쳐서 r의 조절로 규제를 조절함

$$J\left( \theta \right) =MSE\left( \theta \right) +r\alpha \sum _{i=1}^{n}\left| \theta _{i}\right| +\frac {1-r} {2}\alpha \sum _{i=1}^{n}\theta _{i}^{2}$$

참고자료(좋은 글 감사합니다.) : 

[https://ratsgo.github.io/machine learning/2017/05/19/biasvar/](https://ratsgo.github.io/machine%20learning/2017/05/19/biasvar/)

[https://ratsgo.github.io/machine learning/2017/05/22/RLR/](https://ratsgo.github.io/machine%20learning/2017/05/22/RLR/)
