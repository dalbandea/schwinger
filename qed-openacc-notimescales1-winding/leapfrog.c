#include <stdlib.h>
#include <math.h>
#include "hmc.h"
#include "leapfrog.h"
#include "dirac.h"
#include "fields.h"

/*  leap frog */
void leapfrog(const int nsteps, const double dtau) {
  int l;

  /* first phase: \Delta\Tau / 2 step for p */
  update_momenta(0.5*dtau); 

  /*  second phase: iterate with steps of \Delta\Tau */
  for(l = 0; l < nsteps-1; l++) {
    update_gauge(dtau);
    update_momenta(dtau);
  }
  /* a last one for the fields (because N steps for fields, */
  /*      and N-1 steps for impulses) */
  update_gauge(dtau);

  /*  last phase: \Delta\Tau / 2 step for p */
  update_momenta(dtau*0.5);
}

void update_momenta(const double dtau) 
{
  int i;
  g_cgiterations1 += cg(g_X, g_fermion, ITER_MAX, DELTACG, &gam5D_SQR_wilson);
  gam5D_wilson(g_gam5DX, g_X);
#pragma acc parallel loop present(gauge1[0:GRIDPOINTS], gauge2[0:GRIDPOINTS], link1[0:GRIDPOINTS], link2[0:GRIDPOINTS], right1[0:GRIDPOINTS], right2[0:GRIDPOINTS], left1[0:GRIDPOINTS], left2[0:GRIDPOINTS], gp1[0:GRIDPOINTS], gp2[0:GRIDPOINTS], g_X[0:GRIDPOINTS], g_gam5DX[0:GRIDPOINTS]) 
  for(i = 0; i < GRIDPOINTS; i++) {
    /* gp1[i] = gp1[i] - dtau*(DS_G1(i) - trX_dQ_wilson_dalpha1_X(i)); */
    gp1[i] = gp1[i] - dtau*( (beta*(-sin(gauge1[left2[i]]+gauge2[right1[left2[i]]]-gauge1[i]-gauge2[left2[i]])
		+sin(gauge1[i]+gauge2[right1[i]]-gauge1[right2[i]]-gauge2[i]))) - (creal(I*((cconj(link1[i]) 
		   * (cconj(g_X[right1[i]].s1)*(g_R*g_gam5DX[i].s1 +  g_gam5DX[i].s2) -
		      cconj(g_X[right1[i]].s2)*(  g_gam5DX[i].s1   + g_R*g_gam5DX[i].s2))) 
		  -
		  (link1[i] 
		   * (cconj(g_X[i].s1) * (g_R*g_gam5DX[right1[i]].s1-  g_gam5DX[right1[i]].s2) + 
		      cconj(g_X[i].s2) * (  g_gam5DX[right1[i]].s1-g_R*g_gam5DX[right1[i]].s2)))
		  )
	       )) );
    /* gp2[i] = gp2[i] - dtau*(DS_G2(i) - trX_dQ_wilson_dalpha2_X(i)); */
    gp2[i] = gp2[i] - dtau*( (beta*(sin(gauge1[left1[i]]+gauge2[i]-gauge1[left1[right2[i]]]-gauge2[left1[i]])
		-sin(gauge1[i]+gauge2[right1[i]]-gauge1[right2[i]]-gauge2[i]))) - (creal(I*((cconj(link2[i]) 
		   * (cconj(g_X[right2[i]].s1)*(g_R*g_gam5DX[i].s1  - I*g_gam5DX[i].s2) -
		      cconj(g_X[right2[i]].s2)*(I*g_gam5DX[i].s1    + g_R*g_gam5DX[i].s2))) 
		  -
		  (link2[i] 
		   * (cconj(g_X[i].s1) * (g_R*g_gam5DX[right2[i]].s1+I*g_gam5DX[right2[i]].s2) +
		      cconj(g_X[i].s2) * (I*g_gam5DX[right2[i]].s1-g_R*g_gam5DX[right2[i]].s2)))
		  )
	       )) );
  }
  return;
}

void update_gauge(const double dtau) {
  int i;
/* #pragma acc kernels */
#pragma acc parallel loop present(gauge1[0:GRIDPOINTS], gauge2[0:GRIDPOINTS], gp1[0:GRIDPOINTS], gp2[0:GRIDPOINTS]) copyin(dtau)
  for(i = 0; i < GRIDPOINTS; i++) {
    gauge1[i] = gauge1[i] + dtau*gp1[i];
    gauge2[i] = gauge2[i] + dtau*gp2[i];
  }
  calculatelinkvars();
  return;
}
