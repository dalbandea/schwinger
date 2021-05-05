#include <stdlib.h>
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
  for(i = 0; i < GRIDPOINTS; i++) {
    gp1[i] = gp1[i] - dtau*DS_G1(i);
    gp2[i] = gp2[i] - dtau*DS_G2(i);
  }
  return;
}

void update_gauge(const double dtau) {
  int i;
  for(i = 0; i < GRIDPOINTS; i++) {
    gauge1[i] = gauge1[i] + dtau*gp1[i];
    gauge2[i] = gauge2[i] + dtau*gp2[i];
  }
  calculatelinkvars();
  return;
}
