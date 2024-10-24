from dolfin import *
from dolfin.cpp.mesh import *
from mshr import *
import ufl_legacy as ufl
from ufl_legacy import tanh

def W0(u):
    eps = variable(sym(grad(u)))
    e11, e12, e22 = eps[0,0], eps[0,1], eps[1,1]
    E = 0.5*c1111*e11**2 + 0.5*c2222*e22**2 + 2*c2121*e12**2 + 2*c1112*e12*e11 +\
        2*c2212*e12*e22 + c1122*e11*e22
    stress = diff(E, eps) 
    return [E, stress]


def W1(u,d): 
    eps = variable(sym(grad(u)))
    e11, e12, e22 = eps[0,0], eps[0,1], eps[1,1]
    d= variable(d)           
    
    #n_1 ,n_2 =  d[0]/(mag(d)) ,  d[1]/(mag(d))
    n_1 ,n_2 =  d[0]/(sqrt(dot(d,d))) ,  d[1]/(sqrt(dot(d,d)))
    e11_n = e22*n_1**2 -2*e12*n_1*n_2 + e11*n_2**2
    e22_n = e11*n_1**2 + 2*e12*n_1*n_2 + e22*n_2**2 

    n1 ,n2 =  d[0]/(sqrt(dot(d,d))) ,  d[1]/(sqrt(dot(d,d)))


    c_tttt = n2**4 *c1111 + n1**4 *c2222 + 4*n1**2 *n2**2*c2121 \
        -4*n2**3*n1*c1112 -4*n1**3*n2*c2212 + 2*n1**2*n2**2 *c1122 
        
    c_tttn = n2**3*n1*c1111 -n1**3*n2*c2222 + 2*n1**3*n2*c2121 -2* n2**3*n1*c2121 \
        +(n2**4-3*n1**2*n2**2)*c1112 + (-n1**4 + 3*n1**2*n2**2)*c2212 \
           + (n1**3 *n2 -n1*n2**3)*c1122  
        
        
    c_nnnn = n1**4*c1111 + n2**4*c2222 + 4*n1**2*n2**2 *c2121 \
        +4*n1**3*n2*c1112 + 4*n1*n2**3*c2212 \
            + 2*n1**2*n2**2 *c1122
            
    c_ttnn = n1**2*n2**2*c1111 + n1**2*n2**2*c2222 -4* n1**2*n2**2*c2121 \
        +(2*n2**3*n1-2*n2*n1**3)*c1112 + (2*n1**3*n2 - 2*n1*n2**3)*c2212 \
            + (n2**4 + n1**4) *c1122 
    c_tnnn = n2*n1**3*c1111 - n1*n2**3*c2222 -2*n1**3*n2*c2121 + 2*n2**3*n1*c2121 \
        +(3*n1**2 * n2**2- n1**4)*c1112 + (-3*n1**2 * n2**2 + n2**4)*c2212 \
            + (n1 *n2**3 - n2 *n1**3) *c1122
        
    c_tntn = n1**2*n2**2*c1111 + n1**2*n2**2*c2222 + n1**4*c2121 + n2**4*c2121 \
        -2* n1**2*n2**2*c2121 \
            +(2*n1*n2**3 - 2*n2*n1**3)*c1112 + (2*n2*n1**3 - 2*n1*n2**3)*c2212 \
                -2*n1**2 *n2**2*c1122
    
    energy1 = 0.5*(c_tttt - (c_tttn**2*c_nnnn - 2*c_tttn*c_ttnn*c_tnnn +c_ttnn**2*c_tntn)\
                   /(c_tntn * c_nnnn - c_tnnn**2)) *e11_n**2
    energy2 = 0.5 * (c_tttt -c_tttn**2/c_tntn)*e11_n**2 +\
        (c_ttnn - c_tttn*c_tnnn/c_tntn)* e11_n*e22_n+\
        +0.5*(c_nnnn - c_tnnn**2/ c_tntn)*e22_n**2
    
    
        
    e22_s_n = ((c_tttn*c_tnnn - c_ttnn * c_tntn)/(c_tntn* c_nnnn - c_tnnn**2))\
        *e11_n
    
    E =   conditional(lt(e22_n, e22_s_n),  energy2, \
          energy1 )                             
    stress, dE_dd = diff(E, eps) , diff(E, d) 
    return [E, stress, dE_dd]

def fourth_and_second_tensor(u):
    I = Identity(len(u))
    a = Constant((0.7071,  0.7071)) #Constant((1, 0))  #Constant((1, 0)) #given structural director
    M = outer(a,a)
    alpha = 200 #200 
    alpha1 = 200 #200 
    A_second_order =  (I + alpha * M )
    i,j,k,l = ufl.indices(4)
    dim_mesh = mesh.geometry().dim()
    delta = Identity(dim_mesh)
    Identity_fourth = as_tensor( 0.5*(delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k] ), (i,j,k,l) )
    A_fourth_order = (Identity_fourth + alpha1 * outer(M,M) )
    return [A_second_order, A_fourth_order] 

def anisotropic_surface_energy(u,phi, grad_phi):
    
    A_second = fourth_and_second_tensor(u)[0]
    A_fourth = fourth_and_second_tensor(u)[1]
    Gamma_d = Gc* ( dot(phi, phi)/(2*l) + \
        (l/4)*inner(A_second, outer(grad(phi), grad(phi)))   + \
        (l**3/32)*  inner(A_fourth , outer( grad(grad_phi),  grad(grad_phi)) ) )
        #1*1.e20*(l/2)*inner(A_second, grad(grad_phi))  
    return Gamma_d
        

def total_energy(u,phi,d, phi_prev, grad_phi): #, grad_phi
    E = ((1- phi)**2 + eta_eps)*W0(u)[0] +\
        (1-(1- phi)**2 )*\
        conditional(lt(dot(d,d), Cr),0.0, W1(u,d)[0]) +\
        anisotropic_surface_energy(u,phi, grad_phi) +\
        H_l_c(phi_prev, 0.9) * 1.e3*(phi-phi_prev)**2   
    return E

def mag(d):
    return sqrt(dot(d,d) + 1.e-3)

def H_l_c(phi, phi_c):
    Hl = 0.5 + 0.5*tanh((phi-phi_c)/0.001) 
    return Hl


def H_l(phi):
    Hl = 0.5 + 0.5*tanh((phi-0.98)/0.001) 
    return Hl

def H_l_gr(gr_phi):
    gr_phi_mag = mag(gr_phi)
    H_l_gr = 0.5 + 0.5*tanh((gr_phi_mag-10)/la)
    return H_l_gr
    

def E_helm(phi,d, dPrev):
    E_helm0 = outer(d,d) - phi*phi*(outer(grad(phi), grad(phi))) /( (mag(grad(phi))**2)) 
    E_helm1 = inner(E_helm0,E_helm0)
    E_helm2 = 0.1*inner(grad(d), grad(d)) 
    E_helm3 = 100*(mag(d)-phi)**2
    E =  1* ( 1.e2 * H_l_gr(grad(phi)) *E_helm1 + 0.8 * H_l(phi) * E_helm3) + 2*1*0.5 *E_helm2 #+ H_l(phi) * 0.8 * E_helm3
    return E


def stress(u,phi,d):
    E = ((1- phi)**2 + eta_eps)*W0(u)[1] +\
        (1-(1- phi)**2 )*\
        conditional(lt(dot(d,d), Cr),stress_null, W1(u,d)[1])
                   
    return E

def max_lam(u,phi,d):
    S_tensor = stress(u,phi,d)
    s11, s12, s21, s22 = S_tensor[0,0], S_tensor[0,1], S_tensor[1,0], S_tensor[1,1]
    Eig_val1 = 0.5 *(s11 + s22 + (s11**2 + 4 *s12*s21 -2*s11*s22 + s22**2)**0.5)
    Eig_val2 = 0.5 *(s11 + s22 - (s11**2 + 4 *s12*s21 -2*s11*s22 + s22**2)**0.5)
    Eig_val_Mx = Max(Eig_val1, Eig_val2)
    v0 = 1
    v1 =  (Eig_val_Mx - s11)/s12
    v0_norm = v0/(sqrt(v0**2+v1**2))
    v1_norm = v1/(sqrt(v0**2+v1**2))
    Eig_vec = Function(W)
    Eig_vec = as_vector([v0_norm, v1_norm])
    return [Eig_val_Mx, Eig_vec,  conditional(lt(Eig_val_Mx, 0),0.0, Eig_val_Mx)*Eig_vec]
    
def Max(a, b): return (a+b+abs(a-b))/Constant(2)

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}   
parameters["form_compiler"]["quadrature_degree"] = 1
# set_log_active(False)

base = Rectangle(Point(-0.5,-1), Point(0.5,1))
hole = Circle(Point(0, 0), 0.05)
mesh = generate_mesh(base, 50) #-hole
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs (p[1]) <= 0.4 :
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs (p[1]) <= 0.4 :
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs (p[1]) <= 0.4 :
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)



W = VectorFunctionSpace(mesh, 'CG', 1)
Wdeg0 = VectorFunctionSpace(mesh, 'DG', 0)
V1 = FunctionSpace(mesh, 'CG', 1)
V0 = FunctionSpace(mesh, 'DG', 0)
u, v, du = Function(W), TestFunction(W), TrialFunction(W)
phi, vphi, dphi = Function(V1), TestFunction(V1), TrialFunction(V1)

unew, uold = Function(W), Function(W)
phinew, phiold = Function(V1), Function(V1)
phi_Prev = Function(V1)
d_Prev= Function(W)

q_gr_phi = TestFunction(W)
TS = TensorFunctionSpace(mesh, 'DG', 0)
TS1 = TensorFunctionSpace(mesh, 'CG', 1)
stress_null = Function(TS)

def top(x, on_boundary):
    return near(x[1], 1) and on_boundary
def bot(x, on_boundary):
    return near(x[1], -1) and on_boundary
def bot_center(x, on_boundary):
    return near(x[0], 0) and near(x[1], -1) #and on_boundary

def left(x, on_boundary):
    return near(x[0], -0.5) and on_boundary

def right(x, on_boundary):
    return near(x[0], 0.5) and on_boundary


F11= Expression("-10*t", t=0, degree = 1) 
F12= 0
F21= 0
F22= Expression("-10*t", t=0, degree = 1) 

u_BC = Expression(("F11*x[0]+F12*x[1]", "F21*x[0]+F22*x[1]"), F11=F11,F12=F12,F21=F21,F22=F22, degree=1)
BC1, BC2 = DirichletBC(W, u_BC, top), DirichletBC(W, u_BC, bot)
BC3, BC4 = DirichletBC(W, u_BC, left) , DirichletBC(W, u_BC, right)                              
bc_u = [BC1, BC2, BC3, BC4]



# side_coef = 0.04 #0.01
# LoadTop = Expression("-3*side_coef*t", t = 0, side_coef=side_coef, degree=1)
# LoadBot = Expression("3*side_coef*t", t = 0, side_coef=side_coef, degree=1)
# bcbot= DirichletBC(W.sub(1), LoadBot, bot)
# bcbot1 = DirichletBC(W.sub(0), Constant(0), bot)
# bctop1= DirichletBC(W.sub(1), LoadTop, top)
# bc_u = [bcbot, bctop1, bcbot1] 
# LoadLeft = Expression("t*side_coef", t = 0, side_coef= side_coef, degree=1)
# LoadRight = Expression("-t*side_coef", t = 0, side_coef=side_coef, degree=1)


boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
AutoSubDomain(top).mark(boundary_subdomains, 1)
dss = ds(subdomain_data=boundary_subdomains)


bc_phi = []
bc_d=[]

d, vd, dd = Function(W), TestFunction(W), TrialFunction(W)
dold, dnew = Function(W), Function(W)
class InitialConditionVec(UserExpression):
    def eval_cell(self, value, x, ufl_cell):
        if abs (x[0] ) <= 0.01 and abs(x[1])<= 0.1:
            value[0] = 1
            value[1] = 0 
        else:
            value[0] = 0.0
            value[1] = 0.0         
    def value_shape(self):
        return (2,)

d.interpolate(InitialConditionVec())
dold.interpolate(InitialConditionVec())
dnew.interpolate(InitialConditionVec())
d_Prev.interpolate(InitialConditionVec())

class InitialConditionScal(UserExpression):
    def eval_cell(self, value, x, ufl_cell):      
        if abs (x[0] ) <= 0.01 and abs(x[1])<= 0.1:
            value[0] = 1
        else:
            value[0] = 0.0               

phiold.interpolate(InitialConditionScal())
phi.interpolate(InitialConditionScal())
phinew.interpolate(InitialConditionScal())
phi_Prev.interpolate(InitialConditionScal())


C1111=  17842222221.555557
C2222=  17842222221.555557
C1122=  6268888889.555557
C1212=  6268888888.88889
C1112=  482222222.2222223
C2212=  482222222.2222223

max_c = (max(C1111, C2222, C1122, C1212, C1112, C2212))/1.5
c1111 =  C1111 /max_c
c2222 =  C2222 /max_c
c1122 = C1122 /max_c
c1212 =  C1212 /max_c
c1112 = C1112 /max_c
c2212 = C2212 /max_c
c2121 = c1212

Gc =  2*4e4 /max_c #1*1.e-7           
l, eta_eps = 0.01, 1.e-3
Cr = 1.e-3

la = 0.01

grad_phi = Function(W)
grad_phi1 = Function(W)
grad_phi = project(grad(phi), W)
        
Pi1 = total_energy(u, phiold, dold, phi_Prev, grad_phi) * dx   #\
   #- (0)* LoadLeft *u[0]*dss(1) -(0)* LoadRight *u[0]*dss(2)       					    
E_du = derivative(Pi1, u, v)   
J_u = derivative(E_du, u, du) 
p_disp = NonlinearVariationalProblem(E_du, u, bc_u, J_u)
solver_disp = NonlinearVariationalSolver(p_disp)

Pi2 = total_energy(unew, phi, dold, phi_Prev, grad_phi1) * dx #\
    #-  (0)* LoadLeft *u[0]*dss(1) - (0)* LoadRight *u[0]*dss(2) 
E_phi = derivative(Pi2, phi, vphi) 
J_phi  = derivative(E_phi, phi, dphi)   
p_phi = NonlinearVariationalProblem(E_phi, phi, bc_phi ,J_phi)
solver_phi = NonlinearVariationalSolver(p_phi)

Pi3 = E_helm(phinew,d, d_Prev) * dx          					    
E_dd = derivative(Pi3, d, vd)   
J_d = derivative(E_dd, d, dd)    
Prob_d = NonlinearVariationalProblem(E_dd, d, bc_d, J_d)
solver_d = NonlinearVariationalSolver(Prob_d)


Eq_grad_phi = inner(grad_phi1, q_gr_phi)*dx - inner(grad(phi), q_gr_phi)*dx
bc_gr_phi = []


prm1 = solver_disp.parameters
prm1['newton_solver']['maximum_iterations'] = 1000
prm1['newton_solver']['absolute_tolerance'] = 1e-8
prm1['newton_solver']['relative_tolerance'] = 1e-6

prm2 = solver_phi.parameters
prm2['newton_solver']['maximum_iterations'] = 1000
prm2['newton_solver']['absolute_tolerance'] = 1e-8
prm2['newton_solver']['relative_tolerance'] = 1e-6

prm3 = solver_d.parameters
prm3['newton_solver']['maximum_iterations'] = 1000
prm3['newton_solver']['absolute_tolerance'] = 1e-8
prm3['newton_solver']['relative_tolerance'] = 1e-6

CrackScal_file = File ("./Result/crack_scalar.pvd")
CrackVec_file = File ("./Result/crack_Vec.pvd")
Displacement_file = File ("./Result/displacement.pvd") 
EigenVec_file = File ("./Result/EigenVec.pvd")  
CrackVecDeg0_file = File ("./Result/CrackVecDeg0.pvd")  
EigVec = Function(Wdeg0) #max_lam(u,d)[2]
grad_phi_file =  File ("./Result/Grad_phi.pvd") 
Stress_file =  File ("./Result/stress_tot.pvd")
Stress_tot = Function(TS)
prm1['newton_solver']['relaxation_parameter'] = 0.3
prm3['newton_solver']['relaxation_parameter'] = 0.5


phinew.rename("phi", "crack_scalar")
CrackScal_file << phinew
CrackVec_file << dnew
                
                
t = 0
u_r = 2
deltaT  = 1.e-3
tol = 1e-3
Stress_tot = Function(TS)
CrackVecDeg0 = Function(Wdeg0)
ax_coef = 6
               
normal_e2 = Constant((0,1))
               
with open('displacement.txt', 'w') as disp_file, open('traction.txt', 'w') as trac_file:

    
    while t<= 0.22:
        if   t>=0.0135:
        #     deltaT = 1.e-4
             tol = 0.02
        
            
        t += deltaT
        F11.t , F22.t = t, t
        # LoadTop.t, LoadBot.t, LoadLeft.t, LoadRight.t = ax_coef*t*u_r, ax_coef*t*u_r, t*u_r, t*u_r
        iter = 0
        err = 1
        while err > tol:
            iter += 1
            print("solving displacement")
            solver_disp.solve()
            unew.assign(u) 
            print("solving grad phi")
            
            solve(Eq_grad_phi == 0, grad_phi1, bc_gr_phi , solver_parameters= {"newton_solver":{"linear_solver": "mumps", "maximum_iterations": 400}})
            
            print("solving phi")
            
            solver_phi.solve()
            phinew.assign(phi)
            
            print("solving d")
            solver_d.solve()
            dnew.assign(d)
            
            err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
            err_phi = errornorm(phinew,phiold,norm_type = 'l2',mesh = None)
            err_d = errornorm(dnew,dold,norm_type = 'l2',mesh = None)
            err_grad_phi = errornorm(grad_phi1,grad_phi,norm_type = 'l2',mesh = None)
            err = max(err_u,err_phi) #, err_grad_phi
            
            print("error_u =", err_u)
            print("error_phi =", err_phi)
            print("error_d =", err_d)
            
            grad_phi.assign(grad_phi1)
            uold.assign(unew)
            phiold.assign(phinew)
            dold.assign(dnew)
            if err <= tol:
                print ('Iterations:', iter, ', Total time', t)    
                phinew.rename("phi", "crack_scalar")
                CrackScal_file << phinew
                Displacement_file << unew 
                CrackVec_file << dnew
                
                phi_Prev.assign(phinew)
                d_Prev.assign(dnew)
                EigVec = project( max_lam(unew, phinew, dnew)[2],Wdeg0)
                
                Stress_tot = project( stress(unew, phinew, dnew), TS)
                # Stress_file << Stress_tot
                
                disp_magnitude = sqrt(dot(unew, unew))
                disp_integrated = assemble(disp_magnitude * dss(1))
                disp_file.write(f"{disp_integrated}\n")
                
                Stress_tot = project(stress(unew,phinew,dnew), TS)
                traction_top = project(dot(Stress_tot, normal_e2), W)
                traction_magnitude = sqrt(dot(traction_top, traction_top))
                traction_integrated = assemble(traction_magnitude * dss(1))
                trac_file.write(f"{traction_integrated}\n")