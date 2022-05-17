#!/usr/bin/env python
# coding: utf-8

# ### Case 3 Karolina Słapek Wiktor Pyka Kacper Szczepanik

# In[1]:


import pandas as pd
import seaborn as sns
from pylab import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from stargazer.stargazer import Stargazer
import sympy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy
from scipy.stats import spearmanr


# In[2]:


sns.set_style(style="darkgrid")


# In[3]:


df = pd.read_csv('income2wampoprzeksztalceniach.csv', sep = ';')
df.head()


# In[4]:


y=df['lwage']
X=df[['age','age_2','edu_non_grad','edu_uni_col_uni_grad','edu_prof_or_phd_grad','marital_status','member_of_labor_union','sex','race','raceXedu_uni_col_uni_grad','sexXedu_uni_col_uni_grad','sexXedu_non_grad']]
X= sm.add_constant(X)


# In[5]:


model9=smf.ols(formula = 'lwage ~ age + age_2 + edu_non_grad + edu_uni_col_uni_grad + edu_prof_or_phd_grad + marital_status + member_of_labor_union + sex + race +raceXedu_uni_col_uni_grad + sexXedu_uni_col_uni_grad +sexXedu_non_grad',data=df).fit()
model9.summary()


# In[6]:


model9R=smf.ols(formula = 'lwage ~ age + age_2 + edu_non_grad + edu_uni_col_uni_grad + edu_prof_or_phd_grad + marital_status + member_of_labor_union + sex + race +raceXedu_uni_col_uni_grad + sexXedu_uni_col_uni_grad +sexXedu_non_grad',data=df).fit(cov_type='HC0')
model9R.summary()


# In[7]:


Stargazer([model9, model9R])


# ## Zadanie 1
# Korelacja rang spearmana

# In[8]:


scipy.stats.spearmanr(model9.fittedvalues,model9.resid)


# Korelacja < 0,2 oznacza że w modelu występuje korelacja nikła pomiędzy wartościmi i resztami

# In[9]:


plt.rc('figure', figsize = (12,8))


# In[10]:


model9R.summary()


# In[11]:


df['residuals'] = model9.resid


# In[12]:


df.head(10)


# In[13]:


df['residuals_2'] = model9.resid*model9.resid


# In[14]:


df.head(10)


# In[15]:


names1 = ['lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']


# ## Zadanie 2
#  Normalny rozkład reszt

# In[16]:


hist(model9.resid, bins = 100, color = 'steelblue', edgecolor = 'black', alpha = 0.5)
show()


# In[17]:


sm.graphics.plot_regress_exog(model9,"age" )
show()


# In[18]:


sm.graphics.plot_regress_exog(model9,"age_2" )
show()


# ### Zadanie 3

# ## Skośność

# In[19]:


model9.resid.skew()


# 
# ### Składnik losowy wydaje się mieć rozklad normalny. Widzimy na wykresie długie ogony rozkładu

# In[20]:


hist(model9.resid, bins = 100, color = 'steelblue', edgecolor = 'black', alpha = 0.5)
show()


# ## Kurtoza

# In[21]:


model9.resid.kurtosis()


# ##  Test Jarque Bera

# In[22]:


K = model9.resid.kurtosis()+3
S = model9.resid.skew()
n = model9.nobs
print(K)
print(S)


# In[23]:


JB_statystyka_test=(n/6)*(S**2+(1/4)*(K-3)**2)
JB_statystyka_test


# In[24]:


JB_krytyczne = scipy.stats.chi2.sf(JB_statystyka_test, 2)
JB_krytyczne


# Statystyka testowa większa od wartości krytycznej odrzucamy hipotezę o rozkładzie normlanym składnika losowego

# ### Zadanie 4
# ### Test reset

# In[25]:


testreset = sms.linear_reset(model9, use_f = True)
testreset


# H0: liniowa forma funkcyjna
# 
# H1: nieliniowa forma funkcyjna
# 
# P-value = 0.00<0.05
# 
# Odrzucamy H0 o liniowości formy funkcyjnej.
# Metody rozwiązania problemu: uchwycenie korelacji, nieliniowości, przekształcenia lub zmiana metody estymacji.

# Ze wzglęgu na to że mogliśmy zauważyć silną asymetrię w rozkładzie dochodów zadecydowaliśmy o zlogarymowaniu zmiennej objaśnianej wage_per_hour(lwage). 
# 
# W trakcie estymowania modelu, wychwyciliśmy nieliniową zależność między wiekiem a logarytmem dochodu. W związku z tym podjeliśmy decyzję o dodaniu zmiennej age_2 będącej kwadratem zmiennej age.
# 
# Usuneliśmy zmienne dotyczące edukacji posiadające wysoką korelację między sobą.
# 
# 

# ### Zadanie 5

# Za pomocą tego wykresu można zweryfikować założenie KRML dotyczące Stałości wariancji reszt.

# Na wykresie możemy zauważyć że mamy doczynienia z heteroskedastycznością.

# In[26]:


scatter(model9.fittedvalues, model9.resid)
axhline(y = 0, color = 'red')
show()


# ## Zadanie 6

# ## Test White 

# In[27]:


testwhite = sms.het_white(model9.resid, model9.model.exog)
testwhite


# In[28]:


lzip = [names1, testwhite]
lzip


# Po przeprowadzeniu testu White'a i odrzuceniu hipotezy zerowej o homoskedastyczności reszt, podjeliśmy decyzję o zastosowaniu w modelu odpornej macierzy wariancji kowariancji White'a

# In[29]:


testwhite1 = sms.het_white(model9R.resid, model9R.model.exog)
testwhite1


# In[30]:


lzip = [names1, testwhite1]
lzip


# ## Test Breuscha Pagana

# In[31]:


testBP = sms.het_breuschpagan(model9.resid, model9.model.exog)
testBP


# In[32]:


lzip = [names1, testBP]
lzip


# Rozkład empiryczny składnika losolowego mniej więcej pokrywa się rozkładem teoretycznym, z odstającymi obserwacjami na po obydwu stronach rozkładu

# In[33]:


sm.qqplot(model9.resid, line = 'r')
show()


# ## Test Breuscha - Godfreya

# In[34]:


test3 = sms.acorr_breusch_godfrey(model9)
test3


# In[35]:


lzip = [names1,test3]
lzip


# Test Breuscha-Godfreya dotyczy weryfikacji założenia o braku autokorelacji reszt. W przypadku niespełniania założeń w modelu macierz wariancji-kowariancji jest obciążona, estymator nieobciążony ale nieefektywny.

# f p-value= 0.057634607489291535>alfa=0.05 a więc brak podstaw do odrzucenia hipotezy zerowej.

# ## Odległości Cook'a

# In[36]:


np.set_printoptions(suppress=True)


# In[37]:


influence = model9.get_influence()


# In[38]:


cooks=influence.cooks_distance


# In[39]:


print(cooks[0])


# In[40]:


summary=influence.summary_frame()


# In[41]:


summary.head(10)


# In[42]:


leverage = summary[["cooks_d","standard_resid","hat_diag"]]
leverage.head(10)


# In[43]:


fig = plt.figure(figsize=(300,200))
sm.graphics.influence_plot(model9, criterion = 'Cooks')
show()


# Jak widzimy na powyżyszym wykresie, w naszych danych mamy sporo obserwacji nietypowych. Pryzkładową obseracją nietypową jest obserwacja 11173 ze względu na dużą odległość cooka i standaryzowane reszty w module >2

# ## Problem zmiennej nieistotnej

# W naszym modelu nie występuje problem zmiennej nieistotnej 
# W przypadku wystąpienia w modelu zmiennej nieistotnej estymator dalej jest nieobciążony, ale ma większą wariancję, niż w modelu, w którym nie występują zmienne nieistotne. Pogarsza to precyzję oszacowań parametrów przy zmiennych istotnych.

# ## Współliniowość

# Zadanie 10
# Wynikiem wystąpienia problemu współliniowości niedokładnej jest mniejsza precyzja oszacowań, którą widać w błędach standardowych. Przyczynia się ona do zmniejszenia statystyki t, zwiększenia p-value oraz zwiększenia przedziałów ufności, przez co moglibyśmy uwzględnić zmienne nieistotne statystycznie do modelu. W celu naprawienia tego problemu warto usunąć zmienną o najwyższym VIF pod warunkiem, że jest to zmienna nieistotna statystycznie, co sprawdzamy metodą od ogólnego do szczególnego.

# In[44]:


variance_inflation_factor(X.values,2)


# In[45]:


variance_inflation_factor(X.values,1)


# In[46]:


vif = pd.DataFrame()


# In[47]:


varif['VIF Factor'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['features'] = X.columns
vif


# Wysoki VIF age i age_2 spowodowany niedokładną współliniowością tych dwóch zmiennych. Zmienne jednak sa istotne statystycznie

# In[ ]:




