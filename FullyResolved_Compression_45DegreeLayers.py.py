from dolfin import *
from dolfin.cpp.mesh import *
from mshr import *
import ufl_legacy as ufl
from ufl_legacy import tanh

def W0(u):
    eps = variable(sym(grad(u)))
    e11, e12, e22 = eps[0,0], eps[0,1], eps[1,1]
    E = (2*mu*(e11**2 +2*e12**2 + e22**2) + lmbda*(e11**2 + 2*e11*e22 + e22**2))/2
    stress = diff(E, eps) 
    return [E, stress]
    
def W1(u,d): 
    eps = variable(sym(grad(u)))
    e11, e12, e22 = eps[0,0], eps[0,1], eps[1,1]
    d= variable(d)           
    n1 ,n2 =  d[0]/(sqrt(dot(d,d))) ,  d[1]/(sqrt(dot(d,d)))
    e11_n = e22*n1**2 -2*e12*n1*n2 + e11*n2**2
    e22_n = e11*n1**2 + 2*e12*n1*n2 + e22*n2**2 
    e11_n_s = -e11_n * lmbda/(lmbda+2*mu)
    E1 = 0.5*(lmbda +2*mu- lmbda**2/(lmbda + 2*mu) ) * e11_n **2
    E2 = 0.5*(lmbda+2*mu)*(e11_n)**2 + lmbda*e11_n*e22_n + 0.5*(lmbda+2*mu)* e22_n **2
    E =   conditional(lt(e22_n, e11_n_s),  E2, E1)                            
    stress, dE_dd = diff(E, eps) , diff(E, d) 
    return [E, stress, dE_dd]

def fourth_and_second_tensor(u):
    I = Identity(len(u))
    a = Constant((0.9659,  0.2588)) #Constant((1, 0))  #Constant((1, 0)) #given structural director
    M = outer(a,a)
    alpha = 0 #10 #200 
    alpha1 = 0 #10 #200 
    A_second_order = I + alpha * M 
    i,j,k,l = ufl.indices(4)
    dim_mesh = mesh.geometry().dim()
    delta = Identity(dim_mesh)
    Identity_fourth = as_tensor( 0.5*(delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k] ), (i,j,k,l) )
    A_fourth_order = Identity_fourth + alpha1 * outer(M,M) 
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
        H_l(phi_prev) * 1.e3*(phi-phi_prev)**2   
    return E

def mag(d):
    return sqrt(dot(d,d) + 1.e-3)

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
    E = ((1- phi)**2 )*W0(u)[1] +\
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
mesh = generate_mesh(base - hole, 100) #-hole
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if sqrt (p[0]**2 + p[1]**2) <= 0.05:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if sqrt (p[0]**2 + p[1]**2) <= 0.1:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if sqrt (p[0]**2 + p[1]**2) <= 0.1:
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

side_coef = 0.04 #0.01
LoadTop = Expression("-3*side_coef*t", t = 0, side_coef=side_coef, degree=1)
LoadBot = Expression("3*side_coef*t", t = 0, side_coef=side_coef, degree=1)
bcbot= DirichletBC(W.sub(1), LoadBot, bot)
bcbot1 = DirichletBC(W.sub(0), Constant(0), bot_center, method="pointwise")
bctop1= DirichletBC(W.sub(1), LoadTop, top)
bc_u = [bcbot, bctop1, bcbot1] 
LoadLeft = Expression("t*side_coef", t = 0, side_coef= side_coef, degree=1)
LoadRight = Expression("-t*side_coef", t = 0, side_coef=side_coef, degree=1)

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
        if x[0]**2 + (x[1])**2 <= 0.003:
            value[0] = (x[0])/(x[0]**2 + (x[1])**2)**0.5
            value[1] = x[1]/(x[0]**2 + (x[1])**2)**0.5 
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
        if x[0]**2 + (x[1])**2 <= 0.003:
            value[0] = 1
        else:
            value[0] = 0.0               

phiold.interpolate(InitialConditionScal())
phi.interpolate(InitialConditionScal())
phinew.interpolate(InitialConditionScal())
phi_Prev.interpolate(InitialConditionScal())


C1111=  17842222222.222225
C2222=  17842222222.222225
C1122=  6268888888.88889
C1212=  6268888888.88889
C1112=  -482222222.2222223
C2212=  -482222222.2222223

max_c = (max(C1111, C2222, C1122, C1212, C1112, C2212))/1.5

nu = 0.25
Y1 , Y2 = 21.7 * 1e9 , 10.85 * 1e9
mu1, mu2 = Y1/(2+2*nu) / max_c, Y2/(2+2*nu) /max_c
GC1 = 4e5 / max_c
GC2 = 4e4 /max_c

class Gc_Subclass(UserExpression):
    def eval_cell(self, value, x, ufl_cell):      
        if abs(x[1] - x[0] +12/9) <= 1/9 or\
            abs(x[1] - x[0] +8/9) <= 1/9 or \
            abs(x[1] - x[0] +4/9) <= 1/9  or\
            abs(x[1] - x[0]) <= 1/9 or \
            abs(x[1] - x[0] -4/9) <= 1/9 or \
            abs(x[1] - x[0] +-8/9) <= 1/9 or\
            abs(x[1] - x[0] +-12/9) <= 1/9:
            value[0] = GC2
        else:
            value[0] = GC1 

Gc = Function(V1)
Gc.interpolate(Gc_Subclass())

Gc_file = File ("./Result/Gc.pvd")
Gc_file << Gc


class Mu_Subclass(UserExpression):
    def eval_cell(self, value, x, ufl_cell):      
        if abs(x[1] - x[0] +12/9) <= 1/9 or\
            abs(x[1] - x[0] +8/9) <= 1/9 or \
            abs(x[1] - x[0] +4/9) <= 1/9  or\
            abs(x[1] - x[0]) <= 1/9 or \
            abs(x[1] - x[0] -4/9) <= 1/9 or \
            abs(x[1] - x[0] +-8/9) <= 1/9 or\
            abs(x[1] - x[0] +-12/9) <= 1/9:
            value[0] = mu2
        else:
            value[0] = mu1 
            
mu = Function(V1)
mu.interpolate(Mu_Subclass())
Mu_file = File ("./Result/Mu.pvd")
Mu_file << mu


lmbda = 2*mu*nu/(1-2*nu)

            
l, eta_eps = 0.04, 1.e-3
Cr = 1.e-3

la = 0.01

grad_phi = Function(W)
grad_phi1 = Function(W)
grad_phi = project(grad(phi), W)
        
Pi1 = total_energy(u, phiold, dold, phi_Prev, grad_phi) * dx  # \
    #-  LoadLeft *u[0]*dss(1) - LoadRight *u[0]*dss(2)       					    
E_du = derivative(Pi1, u, v)   
J_u = derivative(E_du, u, du) 
p_disp = NonlinearVariationalProblem(E_du, u, bc_u, J_u)
solver_disp = NonlinearVariationalSolver(p_disp)

Pi2 = total_energy(unew, phi, dold, phi_Prev, grad_phi1) * dx #\
    #-  LoadLeft *u[0]*dss(1) - LoadRight *u[0]*dss(2) 
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

grad_grad_phi = Function(TS1)


H_l_phi_file =  File ("./Result/H_l_phi.pvd")
H_l_grad_phi_file = File ("./Result/H_l_grad_phi.pvd")


H_l_phi = Function(V0)
H_l_grad_phi = Function(V0)

prm1 = solver_disp.parameters
prm1['newton_solver']['maximum_iterations'] = 1000
prm2 = solver_phi.parameters
prm2['newton_solver']['maximum_iterations'] = 1000
prm3 = solver_d.parameters
prm3['newton_solver']['maximum_iterations'] = 1000

CrackScal_file = File ("./Result/crack_scalar.pvd")
CrackVec_file = File ("./Result/crack_Vec.pvd")
Displacement_file = File ("./Result/displacement.pvd") 
EigenVec_file = File ("./Result/EigenVec.pvd")  
CrackVecDeg0_file = File ("./Result/CrackVecDeg0.pvd")  
EigVec = Function(Wdeg0) #max_lam(u,d)[2]
grad_phi_file =  File ("./Result/Grad_phi.pvd") 
Stress_file =  File ("./Result/stress_tot.pvd")

prm1['newton_solver']['relaxation_parameter'] = 0.3
prm3['newton_solver']['relaxation_parameter'] = 0.5

t = 0
u_r = 2
deltaT  = 1.e-3 #0.01
tol = 1.e-3 #deltaT = 0.01 and tol = 0.001 resulted in the crack growth suddenly all over the domain, even in the first step
Stress_tot = Function(TS)
CrackVecDeg0 = Function(Wdeg0)
ax_coef = 6
               
normal_e2 = Constant((0,1))
               
with open('displacement.txt', 'w') as disp_file, open('traction.txt', 'w') as trac_file:

    
    while t<= 0.006:
        # if   t>=0.0025:
        #     deltaT = 1.e-4
            # tol = 0.01
            
            
        
            
        t += deltaT
        LoadTop.t, LoadBot.t, LoadLeft.t, LoadRight.t = ax_coef*t*u_r, ax_coef*t*u_r, t*u_r, t*u_r
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
                
                # CrackVecDeg0 = project(dnew, Wdeg0)
                # CrackVecDeg0_file << CrackVecDeg0
                
                phi_Prev.assign(phinew)
                d_Prev.assign(dnew)
                EigVec = project( max_lam(unew, phinew, dnew)[2],Wdeg0)
                # EigenVec_file  << EigVec
                
                # grad_grad_phi = project(grad(grad_phi), TS)
                # grad_phi_file << grad_grad_phi
                
                
                disp_magnitude = sqrt(dot(unew, unew))
                disp_integrated = assemble(disp_magnitude * dss(1))
                disp_file.write(f"{disp_integrated}\n")
                
                Stress_tot = project(stress(unew,phinew,dnew), TS)
                traction_top = project(dot(Stress_tot, normal_e2), W)
                traction_magnitude = sqrt(dot(traction_top, traction_top))
                traction_integrated = assemble(traction_magnitude * dss(1))
                trac_file.write(f"{traction_integrated}\n")
