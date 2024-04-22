from dolfin import *
from dolfin.cpp.mesh import *
from mshr import *
import ufl
from ufl import tanh

##################################################################################
#Defining Elasticity and Fracture Functions:

def W0(u): #Elastic energy of intact material
    eps = variable(sym(grad(u)))
    e11, e12, e22 = eps[0,0], eps[0,1], eps[1,1]
    E = 0.5*c1111*e11**2 + 0.5*c2222*e22**2 + 2*c2121*e12**2 + 2*c1112*e12*e11 +\
        2*c2212*e12*e22 + c1122*e11*e22
    stress = diff(E, eps) 
    return [E, stress]


def W1(u,d): #Elastic energy of cracked material
    eps = variable(sym(grad(u)))
    e11, e12, e22 = eps[0,0], eps[0,1], eps[1,1]
    d= variable(d)           
    n_1 ,n_2 =  d[0]/(sqrt(dot(d,d))) ,  d[1]/(sqrt(dot(d,d)))
    e11_n = e22*n_1**2 -2*e12*n_1*n_2 + e11*n_2**2
    e22_n = e11*n_1**2 + 2*e12*n_1*n_2 + e22*n_2**2 
    n1 ,n2 =  d[0]/(sqrt(dot(d,d))) ,  d[1]/(sqrt(dot(d,d)))

    #Anisotropic elasticity tensor components in n1-n2 basis:
    
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
    
    E =   conditional(lt(e22_n, e22_s_n),  energy2, energy1 )                             
    stress, dE_dd = diff(E, eps) , diff(E, d) 
    return [E, stress, dE_dd]

def fourth_and_second_tensor(u):
    I = Identity(len(u))
    a = Constant((0.7071,  0.7071)) ##given structural director
    M = outer(a,a)
    alpha = 100  
    alpha1 = 100  
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
    return Gamma_d
        
def total_energy(u,phi,d, phi_prev, grad_phi): 
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

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}   
parameters["form_compiler"]["quadrature_degree"] = 1

##################################################################################
#Creating the domain and mesh:
class MeshGenerator:
    def __init__(self, base_shape, hole_shape, refinement_criteria):
        self.mesh = self._generate_and_refine_mesh(base_shape, hole_shape, refinement_criteria)

    def _generate_and_refine_mesh(self, base_shape, hole_shape, refinement_criteria):
        domain = base_shape - hole_shape
        mesh = generate_mesh(domain, 100)
        for distance in refinement_criteria:
            cell_markers = MeshFunction("bool", mesh, 2, False)
            for cell in cells(mesh):
                p = cell.midpoint()
                if sqrt(p[0]**2 + p[1]**2) <= distance:
                    cell_markers[cell] = True
            mesh = refine(mesh, cell_markers)
        return mesh

    def get_mesh(self):
        return self.mesh
    
if __name__ == "__main__":
    base = Rectangle(Point(-0.5, -1), Point(0.5, 1))
    hole = Circle(Point(0, 0), 0.05)
    refinement_criteria = [0.05, 0.1, 0.1]  # Distances for mesh refinement

    mesh_generator = MeshGenerator(base, hole, refinement_criteria)
    mesh = mesh_generator.get_mesh()    

##################################################################################
#Defining Function Spaces:
W = VectorFunctionSpace(mesh, 'CG', 1)
V1 = FunctionSpace(mesh, 'CG', 1)
u, v, du = Function(W), TestFunction(W), TrialFunction(W)
phi, vphi, dphi = Function(V1), TestFunction(V1), TrialFunction(V1)

unew, uold = Function(W), Function(W)
phinew, phiold = Function(V1), Function(V1)
phi_Prev = Function(V1)
d_Prev= Function(W)

grad_phi = Function(W)
grad_phi1 = Function(W)
q_gr_phi = TestFunction(W)

##################################################################################
#Defining boundaries of the domain:
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
    
#Defining dirichlet boundary conditions:
side_coef = 0.04 
LoadTop = Expression("-3*side_coef*t", t = 0, side_coef=side_coef, degree=1)
LoadBot = Expression("3*side_coef*t", t = 0, side_coef=side_coef, degree=1)
bcbot= DirichletBC(W.sub(1), LoadBot, bot)
bcbot1 = DirichletBC(W.sub(0), Constant(0), bot_center, method="pointwise")
bctop1= DirichletBC(W.sub(1), LoadTop, top)
bc_u = [bcbot, bctop1, bcbot1] #Dirichlet boundary condition

#Defining traction boundary conditions:
LoadLeft = Expression("t*side_coef", t = 0, side_coef= side_coef, degree=1)
LoadRight = Expression("-t*side_coef", t = 0, side_coef=side_coef, degree=1)
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
AutoSubDomain(left).mark(boundary_subdomains, 1)
AutoSubDomain(right).mark(boundary_subdomains, 2)
dss = ds(subdomain_data=boundary_subdomains)


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
grad_phi = project(grad(phi), W)


#Anisotropic Elasticity Tensor Components:
C1111=  17921.25
C2222=  17921.25
C1122=  5291.25
C1212=  6671.25
C1112=  -397.75
C2212=  -397.75

max_c = (max(C1111, C2222, C1122, C1212, C1112, C2212))/1.5
c1111 =  C1111 /max_c
c2222 =  C2222 /max_c
c1122 =  C1122 /max_c
c1212 =  C1212 /max_c
c1112 =  C1112 /max_c
c2212 =  C2212 /max_c
c2121 = c1212

Gc, l, eta_eps, Cr, la =  1*1.e-7 , 0.04, 1.e-3, 1.e-3, 0.01         


Pi1 = total_energy(u, phiold, dold, phi_Prev, grad_phi) * dx   \
    -  LoadLeft *u[0]*dss(1) - LoadRight *u[0]*dss(2)       					    
E_du = derivative(Pi1, u, v)   
J_u = derivative(E_du, u, du) 
p_disp = NonlinearVariationalProblem(E_du, u, bc_u, J_u)
solver_disp = NonlinearVariationalSolver(p_disp)

bc_phi = []
Pi2 = total_energy(unew, phi, dold, phi_Prev, grad_phi1) * dx \
    -  LoadLeft *u[0]*dss(1) - LoadRight *u[0]*dss(2) 
E_phi = derivative(Pi2, phi, vphi) 
J_phi  = derivative(E_phi, phi, dphi)   
p_phi = NonlinearVariationalProblem(E_phi, phi, bc_phi ,J_phi)
solver_phi = NonlinearVariationalSolver(p_phi)

bc_d=[]
Pi3 = E_helm(phinew,d, d_Prev) * dx          					    
E_dd = derivative(Pi3, d, vd)   
J_d = derivative(E_dd, d, dd)    
Prob_d = NonlinearVariationalProblem(E_dd, d, bc_d, J_d)
solver_d = NonlinearVariationalSolver(Prob_d)

Eq_grad_phi = inner(grad_phi1, q_gr_phi)*dx - inner(grad(phi), q_gr_phi)*dx
bc_gr_phi = []

prm1 = solver_disp.parameters
prm1['newton_solver']['maximum_iterations'] = 1000
prm2 = solver_phi.parameters
prm2['newton_solver']['maximum_iterations'] = 1000
prm3 = solver_d.parameters
prm3['newton_solver']['maximum_iterations'] = 1000

CrackScal_file = File ("./Result/crack_scalar.pvd")
CrackVec_file = File ("./Result/crack_Vec.pvd")
Displacement_file = File ("./Result/displacement.pvd") 

prm1['newton_solver']['relaxation_parameter'] = 0.3
prm3['newton_solver']['relaxation_parameter'] = 0.5

t = 0
u_r = 0.2
deltaT  = 1.e-3
tol = 1e-3
ax_coef = 6
               
with open('times.txt', 'w') as file: 
    while t<= 0.2:
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
            err = max(err_u,err_phi) 
            
            print("error_u =", err_u)
            print("error_phi =", err_phi)
            print("error_d =", err_d)
            file.write("Still solving "+"err_u"+ str(err_u) +\
                "err_phi"+ str(err_phi) +"err_d"+ str(err_d) + "err_grad_d"+ str(err_grad_phi) + '\n')
            
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
                
                file.write("t= "+ str(t)+"err_u"+ str(err_u) +\
                  "err_phi"+ str(err_phi) +"err_d"+ str(err_d) + "err_grad_d" + str(err_grad_phi)+ '\n')
