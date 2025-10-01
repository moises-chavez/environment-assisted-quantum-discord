import numpy as np
from sympy import*
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import integrate


##p.aestuarii trímero eq 51 MEAD
#Ham=np.array([[215.0,-104.1,5.1,-4.3,4.7,-15.1,-7.8], [-104.1,220.0,32.6,7.1,5.4,8.3,0.8],[5.1,32.6,0.0,-46.8,1.0,-8.1,5.1],[-4.3,7.1,-46.8,125.0,-70.7,-14.7,-61.5],[4.7,5.4,1.0,-70.7,450.0,89.7,-2.5],[-15.1,8.3,-8.1,-14.7,89.7,330.0,32.7],[-7.8,0.8,5.1,-61.5,-2.5,32.7,280.0]])

#chlorobium tepidum
Ham=np.array([[200.0,-87.7,5.5,-5.9,6.7,-13.7,-9.9], [-87.7,320.0,30.8,8.2,0.7,11.8,4.3],[5.5,30.8,0.0,-53.5,-2.2,-9.6,6.0],[-5.9,8.2,-53.5,110.0,-70.7,-17.0,-63.3],[6.7,0.7,-2.2,-70.7,270.0,81.1,-1.3],[-13.7,11.8,-9.6,-17.0,81.1,420.0,39.7],[-9.9,4.3,6.0,-63.3,-1.3,39.7,230.0]]) 





Ham1=np.insert(Ham, [0],[ [0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0] ],axis=1)

Ham2=np.insert(Ham1, [8],[ [0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0] ],axis=1)

Ham3=np.insert(Ham2, [0],[0.0],axis=0)

Ham4=np.insert(Ham3, [8],[0.0],axis=0)



H=Ham4*4.55634*10**-6 #/(2.19475*10**5) 





r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48, s49, s50, s51, s52, s53, s54, s55, s56, s57, s58, s59, s60, s61, s62, s63, s64, s65, s66, s67, s68, s69, s70, s71, s72, s73, s74, s75, s76, s77, s78, s79, s80 = symbols('r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 r37 r38 r39 r40 r41 r42 r43 r44 r45 r46 r47 r48 r49 r50 r51 r52 r53 r54 r55 r56 r57 r58 r59 r60 r61 r62 r63 r64 r65 r66 r67 r68 r69 r70 r71 r72 r73 r74 r75 r76 r77 r78 r79 r80 s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 s34 s35 s36 s37 s38 s39 s40 s41 s42 s43 s44 s45 s46 s47 s48 s49 s50 s51 s52 s53 s54 s55 s56 s57 s58 s59 s60 s61 s62 s63 s64 s65 s66 s67 s68 s69 s70 s71 s72 s73 s74 s75 s76 s77 s78 s79 s80')


r=Matrix( [ [r0, r1, r2, r3, r4, r5, r6, r7, r8], [r9, r10, r11, r12, r13, r14, r15, r16, r17], [r18, r19, r20, r21, r22, r23, r24, r25, r26], [r27, r28, r29, r30, r31, r32, r33, r34, r35], [r36, r37, r38, r39, r40, r41, r42, r43, r44], [r45, r46, r47, r48, r49, r50, r51, r52, r53], [r54, r55, r56, r57, r58, r59, r60, r61, r62], [r63, r64, r65, r66, r67, r68, r69, r70, r71], [r72, r73, r74, r75, r76, r77, r78, r79, r80] ])

s=Matrix( [ [s0, s1, s2, s3, s4, s5, s6, s7, s8], [s9, s10, s11, s12, s13, s14, s15, s16, s17], [s18, s19, s20, s21, s22, s23, s24, s25, s26], [s27, s28, s29, s30, s31, s32, s33, s34, s35], [s36, s37, s38, s39, s40, s41, s42, s43, s44], [s45, s46, s47, s48, s49, s50, s51, s52, s53], [s54, s55, s56, s57, s58, s59, s60, s61, s62], [s63, s64, s65, s66, s67, s68, s69, s70, s71], [s72, s73, s74, s75, s76, s77, s78, s79, s80] ])




h=1.0


Hr=H*r
rH=r*H
Hs=H*s
sH=s*H
z1=(1.0/h)*(Hs-sH)
z2=(1.0/h)*(rH-Hr)


beta=0.0
ss1=-2.0*beta*r
ss2=-2.0*beta*s

x1=zeros(9,9)
x2=zeros(9,9)
x1[0,0]=(r0+r10+r20+r30+r40+r50+r60+r70+r80)*2.0*beta
x2[0,0]=(s0+s10+s20+s30+s40+s50+s60+s70+s80)*2.0*beta

u1=zeros(9,9)
u2=zeros(9,9)
gama=0.0
u1[8,8]=2*gama*r[3,3]
u2[8,8]=2*gama*s[3,3]


t1=zeros(9,9)
t2=zeros(9,9)
for i in range(0,9):
        t1[3,i]=r[3,i]*(-gama)
for i in range(0,9):
        t2[3,i]=s[3,i]*(-gama)


v1=zeros(9,9)
v2=zeros(9,9)
for i in range(0,9):
        v1[i,3]=r[i,3]*(-gama)
for i in range(0,9):
        v2[i,3]=s[i,3]*(-gama)



unit1=z1
unit2=z2
diss1=x1+ss1
diss2=x2+ss2
sink1=u1+t1+v1
sink2=u2+t2+v2


a=0.0
b=7500.0*10**-15/(2.41888*10**-17)
c=100
T=np.linspace(a,b,c)


def regla(valor):
        x=0
        if valor==0 :
                x=0
        elif valor==1:
                x=2
        elif valor==2:
                x=1
        elif valor==3:
                x=4
        elif valor==4:
                x=8
        elif valor==5:
                x=16
        elif valor==6:
                x=32
        elif valor==7:
                x=64
        elif valor==8:
                x=128
        return x

Sx=np.array([[0,1],[1,0]])
Sy=np.array([[0,1j],[-1j,0]])
Sz=np.array([[1,0],[0,-1]])

I=np.array([[1,0],[0,1]])

#alfa=np.array( [ au*0.7*10**-4 ] )

#alfa=7.26*10**-5

#alfa=np.array( [au*7.26*10**-9 , au*7.26*10**-8.5 , au*7.26*10**-8 , au*7.26*10**-7.5 , au*7.26*10**-7 , au*7.26*10**-6.5 , au*7.26*10**-6 , au*7.26*10**-5.5 , au*7.26*10**-5 , au*7.26*10**-4.5 , au*7.26*10**-4 , au*7.26*10**-3.5 , au*7.26*10**-3 , au*7.26*10**-2.5 , au*7.26*10**-2 , au*7.26*10**-1.5 , au*7.26*10**-1] )


alfa = np.array( [ 1*10**-10, 1*10**-8, 1*10**-6, 1*10**-4, 1*10**-2, 1*10**-1, 1*10**0, 1*10**1, 1*10**2, 1*10**4 ])


lqu_acumulada_t_final=np.zeros((7,len(alfa)))

lqu = np.zeros((7,c,len(alfa)))

lqu_integrada = np.zeros((7,c,len(alfa)))

eficiencia_t_final=np.zeros((2,len(alfa)))


ground_t_final=np.zeros(len(alfa))
sitio1_t_final=np.zeros(len(alfa))
sitio2_t_final=np.zeros(len(alfa))
sitio3_t_final=np.zeros(len(alfa))
sitio4_t_final=np.zeros(len(alfa))
sitio5_t_final=np.zeros(len(alfa))
sitio6_t_final=np.zeros(len(alfa))
sitio7_t_final=np.zeros(len(alfa))



for j in range(len(alfa)):

        w1=-2.0*alfa[j]*r
        w2=-2.0*alfa[j]*s

        y1=2*alfa[j]*diag(r0,r10,r20,r30,r40,r50,r60,r70,r80)
        y2=2*alfa[j]*diag(s0,s10,s20,s30,s40,s50,s60,s70,s80)

        deph1=y1+w1
        deph2=y2+w2

        Real=unit1+diss1+deph1+sink1
        Imag=unit2+diss2+deph2+sink2


        todos=Real.col_join(Imag)
        linea=flatten(todos)



        init=flatten(zeros(18,9))
        init[60]=1.0

#        init[10]=0.5

#        init[11]=0.5
#        init[19]=0.5
#        init[20]=0.5
#        init[60]=1.0










        g = lambdify((r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48, s49, s50, s51, s52, s53, s54, s55, s56, s57, s58, s59, s60, s61, s62, s63, s64, s65, s66, s67, s68, s69, s70, s71, s72, s73, s74, s75, s76, s77, s78, s79, s80),linea)


        def matriz(arreglo,T,h,alfa,beta,gama):
                r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48, s49, s50, s51, s52, s53, s54, s55, s56, s57, s58, s59, s60, s61, s62, s63, s64, s65, s66, s67, s68, s69, s70, s71, s72, s73, s74, s75, s76, s77, s78, s79, s80=arreglo
                derivada=g(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48, s49, s50, s51, s52, s53, s54, s55, s56, s57, s58, s59, s60, s61, s62, s63, s64, s65, s66, s67, s68, s69, s70, s71, s72, s73, s74, s75, s76, s77, s78, s79, s80)
                return derivada


        sol = odeint(matriz, init, T, args=(h,alfa,beta,gama))

        poblaciones=np.zeros((c,10))
        for i in range(c):
                poblaciones[i,0]=(sol[i,0]**2.0+sol[i,81]**2.0)**0.5
                poblaciones[i,1]=(sol[i,10]**2.0+sol[i,91]**2.0)**0.5
                poblaciones[i,2]=(sol[i,20]**2.0+sol[i,101]**2.0)**0.5
                poblaciones[i,3]=(sol[i,30]**2.0+sol[i,111]**2.0)**0.5
                poblaciones[i,4]=(sol[i,40]**2.0+sol[i,121]**2.0)**0.5
                poblaciones[i,5]=(sol[i,50]**2.0+sol[i,131]**2.0)**0.5
                poblaciones[i,6]=(sol[i,60]**2.0+sol[i,141]**2.0)**0.5
                poblaciones[i,7]=(sol[i,70]**2.0+sol[i,151]**2.0)**0.5
                poblaciones[i,8]=(sol[i,80]**2.0+sol[i,161]**2.0)**0.5


        y_int =2*gama*integrate.cumtrapz(poblaciones[:,3], T, initial=0)

        eficiencia_t_final[0,j]=y_int[c-1]
        eficiencia_t_final[1,j]=poblaciones[c-1,8]
        
        ground_t_final[j]=poblaciones[c-1,0]
        sitio1_t_final[j]=poblaciones[c-1,1]
        sitio2_t_final[j]=poblaciones[c-1,2]
        sitio3_t_final[j]=poblaciones[c-1,3]
        sitio4_t_final[j]=poblaciones[c-1,4]
        sitio5_t_final[j]=poblaciones[c-1,5]
        sitio6_t_final[j]=poblaciones[c-1,6]
        sitio7_t_final[j]=poblaciones[c-1,7]

        bb=np.zeros((9,9,c))
        for i in range(0,81):
                if i<9:
                        bb[0,i,:]=sol[:,i]
                elif i<18:
                        bb[1,i-9,:]=sol[:,i]
                elif i<27:
                        bb[2,i-18,:]=sol[:,i]
                elif i<36:
                        bb[3,i-27,:]=sol[:,i]
                elif i<45:
                        bb[4,i-36,:]=sol[:,i]
                elif i<54:
                        bb[5,i-45,:]=sol[:,i]
                elif i<63:
                        bb[6,i-54,:]=sol[:,i]
                elif i<72:
                        bb[7,i-63,:]=sol[:,i]
                else:
                        bb[8,i-72,:]=sol[:,i]


        aa=np.zeros((9,9,c))
        for i in range(81,162):
                if i<90:
                        aa[0,i-81,:]=sol[:,i]
                elif i<99:
                        aa[1,i-90,:]=sol[:,i]
                elif i<108:
                        aa[2,i-99,:]=sol[:,i]
                elif i<117:
                        aa[3,i-108,:]=sol[:,i]
                elif i<126:
                        aa[4,i-117,:]=sol[:,i]
                elif i<135:
                        aa[5,i-126,:]=sol[:,i]
                elif i<144:
                        aa[6,i-135,:]=sol[:,i]
                elif i<153:
                        aa[7,i-144,:]=sol[:,i]
                else:
                        aa[8,i-153,:]=sol[:,i]


        L=bb+1j*aa



        #rho
        rho= np.zeros((256,256,c),complex)
        for i in range(0,9):
                for ii in range(0,9):
                        rho[regla(i),regla(ii),:]=L[i,ii,:]



        #raizrho
        Z=np.zeros((256,256,c),complex)
        for n in range(0,c):
                U,V=np.linalg.eig(rho[:,:,n])
                for m in range(0,256):
                        Z[:,:,n] = Z[:,:,n] + (U[m]**0.5) * np.outer(V[:,m],np.conjugate(V[:,m]))



        ###################################################################################################
        Px=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,Sx)))))))
        Py=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,Sy)))))))
        Pz=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,Sz)))))))

        W=np.zeros((3,3,c),complex)


        for l in range(0,c):
                W[0,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Px ) ) ) )
                W[0,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Py ) ) ) )
                W[0,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Pz ) ) ) )


                W[1,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Px ) ) ) )
                W[1,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Py ) ) ) )
                W[1,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Pz ) ) ) )

                W[2,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Px ) ) ) )
                W[2,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Py ) ) ) )
                W[2,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Pz ) ) ) )

                UU,VV=np.linalg.eig(W[:,:,l])
                
                lqu[0,l,j]=1.0-max(np.real(UU[0]),np.real(UU[1]),np.real(UU[2]))
        lqu_integrada[0,:,j] = integrate.cumtrapz(lqu[0,:,j], T, initial=0)/b
        lqu_acumulada_t_final[0,j]=lqu_integrada[0,c-1,j]






#################################################################################################
        Px=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(Sx,I)))))))
        Py=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(Sy,I)))))))
        Pz=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(Sz,I)))))))



        #matriz w y lqu

        W=np.zeros((3,3,c),complex)


        for l in range(0,c):
                W[0,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Px ) ) ) )
                W[0,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Py ) ) ) )
                W[0,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Pz ) ) ) )


                W[1,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Px ) ) ) )
                W[1,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Py ) ) ) )
                W[1,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Pz ) ) ) )

                W[2,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Px ) ) ) )
                W[2,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Py ) ) ) )
                W[2,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Pz ) ) ) )

                UU,VV=np.linalg.eig(W[:,:,l])
                

                lqu[1,l,j]=1.0-max(np.real(UU[0]),np.real(UU[1]),np.real(UU[2]))
        lqu_integrada[1,:,j] = integrate.cumtrapz(lqu[1,:,j], T, initial=0)/b
                
        lqu_acumulada_t_final[1,j]=lqu_integrada[1,c-1,j]



        ###################################################################################################
        Px=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(Sx,np.kron(I,I)))))))
        Py=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(Sy,np.kron(I,I)))))))
        Pz=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(Sz,np.kron(I,I)))))))



        #matriz w y lqu

        W=np.zeros((3,3,c),complex)


        for l in range(0,c):
                W[0,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Px ) ) ) )
                W[0,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Py ) ) ) )
                W[0,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Pz ) ) ) )


                W[1,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Px ) ) ) )
                W[1,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Py ) ) ) )
                W[1,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Pz ) ) ) )

                W[2,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Px ) ) ) )
                W[2,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Py ) ) ) )
                W[2,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Pz ) ) ) )

                UU,VV=np.linalg.eig(W[:,:,l])
                

                lqu[2,l,j]=1.0-max(np.real(UU[0]),np.real(UU[1]),np.real(UU[2]))
        lqu_integrada[2,:,j] = integrate.cumtrapz(lqu[2,:,j], T, initial=0)/b
                
        lqu_acumulada_t_final[2,j]=lqu_integrada[2,c-1,j]


        ###################################################################################################
        Px=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(Sx,np.kron(I,np.kron(I,I)))))))
        Py=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(Sy,np.kron(I,np.kron(I,I)))))))
        Pz=np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(Sz,np.kron(I,np.kron(I,I)))))))



        #matriz w y lqu

        W=np.zeros((3,3,c),complex)


        for l in range(0,c):
                W[0,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Px ) ) ) )
                W[0,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Py ) ) ) )
                W[0,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Pz ) ) ) )


                W[1,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Px ) ) ) )
                W[1,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Py ) ) ) )
                W[1,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Pz ) ) ) )

                W[2,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Px ) ) ) )
                W[2,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Py ) ) ) )
                W[2,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Pz ) ) ) )

                UU,VV=np.linalg.eig(W[:,:,l])
                
                
                lqu[3,l,j]=1.0-max(np.real(UU[0]),np.real(UU[1]),np.real(UU[2]))
        lqu_integrada[3,:,j] = integrate.cumtrapz(lqu[3,:,j], T, initial=0)/b
                
        lqu_acumulada_t_final[3,j]=lqu_integrada[3,c-1,j]



        ###################################################################################################
        Px=np.kron(I,np.kron(I,np.kron(I,np.kron(Sx,np.kron(I,np.kron(I,np.kron(I,I)))))))
        Py=np.kron(I,np.kron(I,np.kron(I,np.kron(Sy,np.kron(I,np.kron(I,np.kron(I,I)))))))
        Pz=np.kron(I,np.kron(I,np.kron(I,np.kron(Sz,np.kron(I,np.kron(I,np.kron(I,I)))))))



        #matriz w y lqu

        W=np.zeros((3,3,c),complex)


        for l in range(0,c):
                W[0,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Px ) ) ) )
                W[0,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Py ) ) ) )
                W[0,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Pz ) ) ) )


                W[1,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Px ) ) ) )
                W[1,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Py ) ) ) )
                W[1,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Pz ) ) ) )

                W[2,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Px ) ) ) )
                W[2,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Py ) ) ) )
                W[2,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Pz ) ) ) )

                UU,VV=np.linalg.eig(W[:,:,l])
                
                
                lqu[4,l,j]=1.0-max(np.real(UU[0]),np.real(UU[1]),np.real(UU[2]))
        lqu_integrada[4,:,j] = integrate.cumtrapz(lqu[4,:,j], T, initial=0)/b
                
        lqu_acumulada_t_final[4,j]=lqu_integrada[4,c-1,j]

        ###################################################################################################
        Px=np.kron(I,np.kron(I,np.kron(Sx,np.kron(I,np.kron(I,np.kron(I,np.kron(I,I)))))))
        Py=np.kron(I,np.kron(I,np.kron(Sy,np.kron(I,np.kron(I,np.kron(I,np.kron(I,I)))))))
        Pz=np.kron(I,np.kron(I,np.kron(Sz,np.kron(I,np.kron(I,np.kron(I,np.kron(I,I)))))))



        #matriz w y lqu

        W=np.zeros((3,3,c),complex)


        for l in range(0,c):
                W[0,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Px ) ) ) )
                W[0,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Py ) ) ) )
                W[0,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Pz ) ) ) )


                W[1,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Px ) ) ) )
                W[1,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Py ) ) ) )
                W[1,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Pz ) ) ) )

                W[2,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Px ) ) ) )
                W[2,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Py ) ) ) )
                W[2,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Pz ) ) ) )

                UU,VV=np.linalg.eig(W[:,:,l])
                
                
                lqu[5,l,j]=1.0-max(np.real(UU[0]),np.real(UU[1]),np.real(UU[2]))
        lqu_integrada[5,:,j] = integrate.cumtrapz(lqu[5,:,j], T, initial=0)/b

        lqu_acumulada_t_final[5,j]=lqu_integrada[5,c-1,j]                





        ###################################################################################################
        Px=np.kron(I,np.kron(Sx,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,I)))))))
        Py=np.kron(I,np.kron(Sy,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,I)))))))
        Pz=np.kron(I,np.kron(Sz,np.kron(I,np.kron(I,np.kron(I,np.kron(I,np.kron(I,I)))))))



        #matriz w y lqu

        W=np.zeros((3,3,c),complex)


        for l in range(0,c):
                W[0,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Px ) ) ) )
                W[0,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Py ) ) ) )
                W[0,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Px , np.dot( Z[:,:,l] , Pz ) ) ) )


                W[1,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Px ) ) ) )
                W[1,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Py ) ) ) )
                W[1,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Py , np.dot( Z[:,:,l] , Pz ) ) ) )

                W[2,0,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Px ) ) ) )
                W[2,1,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Py ) ) ) )
                W[2,2,l]=np.trace( np.dot( Z[:,:,l] , np.dot( Pz , np.dot( Z[:,:,l] , Pz ) ) ) )

                UU,VV=np.linalg.eig(W[:,:,l])
                


                lqu[6,l,j]=1.0-max(np.real(UU[0]),np.real(UU[1]),np.real(UU[2]))
                
        lqu_integrada[6,:,j] = integrate.cumtrapz(lqu[6,:,j], T, initial=0)/b
        
        lqu_acumulada_t_final[6,j]=lqu_integrada[6,c-1,j]
        
        
        print(j)



#for i in range(7):
#        plt.plot(T,lqu_integrada[i,:] )
#        plt.xlabel("T")
#        plt.ylabel("LQU_integrada")
#plt.legend(['1','2','3','4','5','6','7'])
#plt.grid()
#plt.show()


plt.plot(alfa, lqu_acumulada_t_final[0,:], label='lqu1')
plt.plot(alfa, lqu_acumulada_t_final[1,:], label='lqu2')
plt.plot(alfa, lqu_acumulada_t_final[2,:], label='lqu3')
plt.plot(alfa, lqu_acumulada_t_final[3,:], label='lqu4')
plt.plot(alfa, lqu_acumulada_t_final[4,:], label='lqu5')
plt.plot(alfa, lqu_acumulada_t_final[5,:], label='lqu6')
plt.plot(alfa, lqu_acumulada_t_final[6,:], label='lqu7')


plt.plot(alfa,eficiencia_t_final[0,:],label='sitio tres integrado')
plt.plot(alfa,eficiencia_t_final[1,:],label='poblacion sink')

plt.plot(alfa,ground_t_final,label='ground')
plt.plot(alfa,sitio1_t_final,label='sitio1')
plt.plot(alfa,sitio2_t_final,label='sitio2')
plt.plot(alfa,sitio3_t_final,label='sitio3')
plt.plot(alfa,sitio4_t_final,label='sitio4')
plt.plot(alfa,sitio5_t_final,label='sitio5')
plt.plot(alfa,sitio6_t_final,label='sitio6')
plt.plot(alfa,sitio7_t_final,label='sitio7')

plt.grid()
plt.show()

plt.legend(loc='best')
plt.xlabel('alfa')

#for s in range(7):
#        plt.plot(T,lqu[s,:,1])
#plt.show()



#plt.plot(T,lqu_integrada[0,:,9])
#plt.plot(T,lqu_integrada[1,:,9])
#plt.plot(T,lqu_integrada[2,:,9])
#plt.plot(T,lqu_integrada[3,:,9])
#plt.plot(T,lqu_integrada[4,:,9])
#plt.plot(T,lqu_integrada[5,:,9])
#plt.plot(T,lqu_integrada[6,:,9])
#plt.show()


#for s in range(7):
#        plt.plot(T,lqu[s,:,0])
#plt.legend(['1','2','3','4','5','6','7'])
#plt.show
