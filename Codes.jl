#Load Packages

using CSV, DataFrames, CategoricalArrays, HypothesisTests, Statistics, StatsBase, GLM, Gadfly, DataStructures, Distributions,LinearAlgebra, ScikitLearn

#Load Data

df = DataFrame(CSV.File("C:/Users/AG/Desktop/Projects/project 6/h209.csv", header = true))

#Clean
df = filter([:ADAGE42,  :ADSEX42, :ADGENH42, :ADMDVT42, :FTSTU18X, :MIDX, :CANCERDX, :DIABDX_M18, :MARRY18X, :EDUCYR, :HIBPDX, :MCARE18, :MCAID18] => 
     (ADAGE42, ADSEX42, ADGENH42, ADMDVT42, FTSTU18X, MIDX, CANCERDX, DIABDX_M18, MARRY18X, EDUCYR, HIBPDX, MCARE18, MCAID18) -> 
	 ADAGE42 >= 0 && ADSEX42 >= 0 && ADGENH42 >= 0 && ADMDVT42 >= 0 && FTSTU18X >= 0 && MIDX >= 0 && CANCERDX >= 0 && DIABDX_M18 >= 0 && MARRY18X >= 0 && EDUCYR >= 0 && HIBPDX >= 0 && MCARE18 >= 0 && MCAID18 >= 0,
	 df)
     
## Choose Variables
age = df.ADAGE42
sex = df.ADSEX42
geh = df.ADGENH42
dvt = df.ADMDVT42
inc = df.FAMINC18
pinc = df.TTLP18X
siz = df.FAMSZE18
stu = df.FTSTU18X
ha = df.MIDX
cancer = df.CANCERDX
dia = df.DIABDX_M18
wrace = df.RACEWX
brace = df.RACEBX
arace = df.RACEAX
hrace = df.HISPNCAT
orace = df.RACETHX
edu = df.EDUCYR
mar = df.MARRY18X
hbp = df.HIBPDX
ins = df.INSURC18
den = df.DVTEXP18
mer = df.MCARE18
mca = df.MCAID18

##Cleam Data
data = DataFrame(age = age, sex = sex, geh = geh, dvt = dvt, inc = inc, pinc = pinc, siz = siz, stu = stu, ha = ha, can = cancer, dia = dia, hbp = hbp
     , wr = wrace, br = brace, ar = arace, hrace = hrace, orace = orace, edu = edu, mar = mar, ins = ins, den = den, mer = mer, mca = mca)

#Models and Tests

model1 = lm(@formula(siz ~ ha + hbp + dia + can), data)
residual1 = data.siz - GLM.predict(model1)
m1 = DataFrame(fitted = GLM.predict(model1), residual = residual1)
plot(m1, x=m1.fitted, y=m1.residual, Geom.point)

model2 = lm(@formula(inc ~ ha + hbp + dia + can), data)
residual2 = data.inc - GLM.predict(model2)
m2 = DataFrame(fitted = GLM.predict(model2), residual = residual2)
plot(m2, x=m2.fitted, y=m2.residual, Geom.point)

model3 = lm(@formula(ins ~ ha + hbp + dia + can), data)
residual3 = data.ins - GLM.predict(model3)
m3 = DataFrame(fitted = GLM.predict(model3), residual = residual3)
plot(m3, x=m3.fitted, y=m3.residual, Geom.point)

model4 = lm(@formula(age ~ ha + hbp + dia + can), data)
residual4 = data.age - GLM.predict(model4)
m4 = DataFrame(fitted = GLM.predict(model4), residual = residual4)
plot(m4, x=m4.fitted, y=m4.residual, Geom.point)

model5 = lm(@formula(inc ~ ins + mer + mca), data)
residual5 = data.inc - GLM.predict(model5)
m5 = DataFrame(fitted = GLM.predict(model5), residual = residual5)
plot(m5, x=m5.fitted, y=m5.residual, Geom.point)

model6 = lm(@formula(mca ~ wr + br + ar), data)
residual6 = data.mca - GLM.predict(model6)
m6 = DataFrame(fitted = GLM.predict(model6), residual = residual6)
plot(m6, x=m6.fitted, y=m6.residual, Geom.point)

model7 = lm(@formula(hbp ~ age + edu + ins + inc + wr + br + ar), data)
residual7 = data.hbp - GLM.predict(model7)
m7 = DataFrame(fitted = GLM.predict(model7), residual = residual7)
plot(m7, x=m7.fitted, y=m7.residual, Geom.point)

model8 = lm(@formula(sex ~ ha + hbp + dia + can), data)
residual8 = data.sex - GLM.predict(model8)
m8 = DataFrame(fitted = GLM.predict(model8), residual = residual8)
plot(m8, x=m8.fitted, y=m8.residual, Geom.point)

model9 = lm(@formula(pinc ~ ins + geh + dvt), data)
residual9 = data.pinc - GLM.predict(model9)
m9 = DataFrame(fitted = GLM.predict(model9), residual = residual9)
plot(m9, x=m9.fitted, y=m9.residual, Geom.point)

##AIC
aic.((model1, model2, model3, model4, model5, model6, model7, model8, model9))

##JB Test
JarqueBeraTest(residual2)
