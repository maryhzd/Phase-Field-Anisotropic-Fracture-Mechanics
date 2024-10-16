import numpy
Y1, nu1 = 21700, 0.25
Y2, nu2 = 10850, 0.25

def Param(Y,nu):
    Mu = Y/(2+2*nu)
    lmbda = 2*Mu*nu/(1-2*nu)
    k = (2/3)*Mu + lmbda
    H = k+(4/3)*Mu
    return [Mu, H]
    
mu1, H1 = Param(Y1,nu1)[0], Param(Y1,nu1)[1]
mu2, H2 = Param(Y2,nu2)[0], Param(Y2,nu2)[1]
t1 , t2 = 0.5, 0.5

def VoigtAvg(a1, a2):
    V_avg = a1*t1 + a2*t2
    return V_avg

c11 = (VoigtAvg(1/H1, 1/H2))**-1
c12 = ( (VoigtAvg(1/H1, 1/H2))**-1 )* VoigtAvg((H1-2*mu1)/H1, (H2-2*mu2)/H2)
c13 = ( (VoigtAvg(1/H1, 1/H2))**-1 )* (VoigtAvg((H1-2*mu1)/H1, (H2-2*mu2)/H2))**2 + \
    2*VoigtAvg((H1-2*mu1)*mu1/H1, (H2-2*mu2)*mu2/H2)
    
c44 = (VoigtAvg(1/mu1, 1/mu2))**-1
c66 = VoigtAvg(mu1, mu2)
c33 = c13 + 2*c66

c1111 = c11
c1122 = c12
c2222 = c33
c1212 = c44 

print(c44)
