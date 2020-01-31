---
layout: post
title: "Regression Analysis(회귀분석)"
subtitle: "공부하기"
categories: study
tags: statistics
comments: true
---

# 회귀분석(Regression Analysis)

### 선형회귀분석(종속변수가 연속형인 경우)

<img src="/data/img/2020-01-31/reg (1).png">

**회귀분석 정의** 

관찰된 **연속형** 변수들에 대해 두 변수들 사이의 모형을 구한 뒤 적합도를 측정해 내는 분석방법 

**단순회귀분석 :** 종속변수 1, 독립변수1 

**다중회귀분석** : 종속변수 1, 독립변수 多

<img src="/data/img/2020-01-31/reg (2).png">

<img src="/data/img/2020-01-31/reg (3).png">

**회귀분석의 표준 가정**

1. 선형성 : 오차항은 모든 독립변수 값에 대하여 동일 분산을 가짐 
2. 정규성 : 오차항의 평균(기대값)은 0, 확률분포는 정규분포 
3. 독립성 : 독립변수 상호 간에는 상관관계가 없어야 함 
4. 시간에 따라 수집한 데이터들은 잡음의 영향을 받지 않아야 함 

    import statsmodels.api as sm;
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv('house_prices.csv')
    df.head()
    
    
    def get_model(seed):
        df_train, df_test = train_test_split(df, test_size=0.5, random_state=seed)
        model = sm.OLS.from_formula("TOTEMP ~ GNPDEFL + POP + GNP + YEAR + ARMED + UNEMP", data=df_train)
        return df_train, df_test, model.fit()
    
    
    df_train, df_test, result = get_model(3)
    print(result.summary())

<img src="/data/img/2020-01-31/reg (4).png">

→ Durbin-Watson(더빈왓슨, DW검정)은 잔차의 독립성을 확인할 수 있는 수치이다. 0이면 잔차들이 양의 자기상관을 갖고, 2이면 자기상관이 없는 독립성을 갖고, 4이면 잔차들이 음의 자기상관을 갖는다고 해석한다. 보통 1.5 ~ 2.5사이이면 독립으로 판단하고 회귀모형이 적합하다는 것을 의미한다. DW검정값이 0 또는 4에 가깝다는 것은 잔차들이 자기상관을 가지고 있다는 의미이고, 이는 t값, F값, R제곱을 실제보다 증가시켜 실제로 유의미하지 않은 결과를 유의미한 결과로 왜곡하게 된다.

→ 위 처럼 train set과 test set을 나누어 회귀 분석을 수행해 차이가 크면 overfitting 여부 확인 가능 

→ random_state를 부여하지 않으면, 회귀 분석 수행을 할 때마다 다른 결과가 나타날 수 있음. random_state의 숫자는 크게 중요하지 않음. 예)3을 사용했으면, 다음에도 3을 사용하면 같은 sample데이터를 사용해 같은 결과를 냄 

→ 위의 경우 조건수가 크게 나타나 다중공선성 문제가 발생할 수 있다는 결과를 냄 

→ 조정결정계수 (Adj. R-squared)

<img src="/data/img/2020-01-31/reg (5).png">


(참조: [https://datascienceschool.net/view-notebook/a60e97ad90164e07ad236095ca74e657/](https://datascienceschool.net/view-notebook/a60e97ad90164e07ad236095ca74e657/))

**회귀분석에서 조건수가 커지는 경우**

1.변수들의 단위 차이로 인해 숫자의 스케일이 크게 다른 경우 → scaling 

<img src="/data/img/2020-01-31/reg (6).png">

→ 이상치가 많은경우 Robust Scaler를 사용하거나, 이상치 제거 후 사용하는 것이 좋다.

    ##StandardScaler를 MinMaxScaler, MaxAbsScaler, RobustScaler로 이름만 대치해서 사용하면 됨 
    from sklearn.preprocessing import StandardScaler
    standardScaler = StandardScaler()
    print(standardScaler.fit(train_data))
    train_data_standardScaled = standardScaler.transform(train_data)

2. 다중공선성 즉, 독립변수들 간 상관관계가 큰 경우 → 아래에 따로 설명

3. 독립 변수나 종속 변수가 심하게 한쪽으로 치우친 분포를 보이는 경우(왜도)

4. 독립 변수와 종속 변수간의 관계가 곱셈 혹은 나눗셉으로 연결된 경우

→3과 4의 경우 일반적으로 log1p 를 취한후 스케일링 수행 

**다중공선성문제**

독립변수 간 강한 상관관계가 나타나는 문제

**진단 방법**

- 결정계수 R-square 값이 높아 회귀식의 설명력은 높지만 독립변수의 p-value값이 커서 유의하지 않은 경우 독립변수들 간에 높은 상관관계가 있다고 의심할 수 있음
- 독립변수들 간 상관계수를 통해 진단

    cmap = sns.light_palette("darkgray", as_cmap=True)
    sns.heatmap(dfX.corr(), annot=True, cmap=cmap)
    plt.show()

- 분산팽창요인(Variance Inflation Factor;VIF)를 구해 이 값이 10을 넘는다면 다중공선성의 문제가 있다고 판단 가능

    <img src="/data/img/2020-01-31/reg (7).png">


    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(
        dfX.values, i) for i in range(dfX.shape[1])]
    vif["features"] = dfX.columns
    vif

**해결 방법**

- 상황을 파악하여 상관관계가 높은 독립변수가 나타나는 이유를 파악하고 이를 해결함
    - 하나 혹은 일부를 제거 또는 더해 나감
    - 변수를 변형시키거나 새로운 관측치를 이용
    - PCA(Principle Component Analysis)를 이용한 diagonal matrix의 형태로 공선성을 없앰

### 다항 회귀

회귀가 독립변수의 단항식이 아닌 2차, 3차 방정식과 같은 다항식으로 표현한 것을 다항(Polynomial) 회귀라고 함

ex. 

$$y=w_0+w_1*x_1+w_2*x_2+w_3*x_1*x_2+w_4*x_1^2+x_5*x_2^2$$

다항회귀가 비선형 회귀라고 오해하기 쉽지만 다항회귀는 선형회귀 

예를 들어, 위의 식에서 새로운 변수인 Z를 

$$Z=[x_1,x_2,x_1*x_2,x_1^2,x_2^2]$$

로 한다면, 

$$y=w_0+w_1*z_1+w_2*z_2+w_3*z_3+w_4*z_4+w_5*z_5$$

로 표현할 수 있기 때문에, 여전히 선형회귀라고 볼 수 있다.

Scikit learn은 다항회귀 클래스를 명시적으로 제공하고 있지는 않음

다만, PolynomialFeatures 클래스를 통해 피처를 Polynomial 피처로 변환은 가능 

    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    
    X=np.arange(4).reshape(2,2)
    poly=PolynomialFeatures(degree=2)
    poly.fit(X)
    poly_ftr=poly.transform(X)

### **로지스틱 회귀분석(종속변수가 범주형 중에서도 0 또는 1로 분류되는 경우; 연속형일 때도 사용가능)**

<img src="/data/img/2020-01-31/reg (8).png">


종속변수가 성공/실패와 같은 이항분포인 경우 선형회귀로는 fitting이 안됨. 

**로지스틱 함수**

로지스틱함수는 음의 무한대부터 양의 무한대까지의 실수값을 0부터 1사이의 실수값으로 1 대 1 대응시키는 시그모이드 함수 

**Odds(오즈)=μ/1-μ** 

베르누이 시도에서 1이 나올 확률 μ 와 0이 나올 확률 1−μ 의 비율(ratio)을 승산비(odds ratio)라고 한다.

**Logit(로짓) 함수 =log(오즈)**

로짓 함수의 값은 로그 변환에 의해 음의 무한대(−∞)부터 양의 무한대(∞)까지의 값을 가질 수 있다.

로지스틱함수(Logistic function)는 로짓함수의 역함수이므로 음의 무한대(−∞)부터 양의 무한대(∞)까지의 값을 가지는 입력변수를 0부터 1사의 값을 가지는 출력변수로 변환함

    from sklearn.linear_model import LogisticRegression
    
    model_sk = LogisticRegression().fit(X0, y)
    
    xx = np.linspace(-3, 3, 100)
    mu = 1.0/(1 + np.exp(-model_sk.coef_[0][0]*xx - model_sk.intercept_[0]))
    plt.plot(xx, mu)
    plt.scatter(X0, y, c=y, s=100, edgecolor="k", lw=2)
    plt.scatter(X0, model_sk.predict(X0), label=r"$\hat{y}$", marker='s', c=y,
                s=100, edgecolor="k", lw=1, alpha=0.5)
    plt.xlim(-3, 3)
    plt.xlabel("x")
    plt.ylabel(r"$\mu$")
    plt.title(r"$\hat{y}$ = sign $\mu(x)$")
    plt.legend()
    plt.show()
