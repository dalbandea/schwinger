/***********************************************************************
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2011 Nils Christian,
 *               Pavel Buividovic, Carsten Urbach
 *
 * This file is part of a Schwinger code for the Helmholtz summer school
 * 2011 in Dubna
 *
 * this is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * this software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this code.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "rand/ranlxd.h"
#include "rand/gauss.h"
#include "linalg.h"
#include "fields.h"
#include "lattice.h"
#include "dirac.h"
#include "2MN_integrator.h"
#include "leapfrog.h"
#include "leapfrog2.h"
#include "rec_lf_integrator.h"
#include "hmc.h"
#ifndef M_PI
# define M_PI    3.14159265358979323846f
#endif


int R;
int g_cgiterations1;
int g_cgiterations2;

int update() //Basic HMC update step
{
  double squnrm;
  int i, acc;
  double exphdiff;
  
  /* the new impulses and the 'generator' of the arbitrary pseudofield */
  /* calculate the hamiltonian of this state: new impulses + action */
  /* g_X is ab-used a bit - here it is \xi = (gamma5 D)^{-1} \phi */
  
  ham_old = s_g_old;

#pragma parallel for shared(gp1, gp2) reduction(+: ham_old)
  for(i=0; i<GRIDPOINTS; i++) {
    gp1[i] = gauss();
    gp2[i] = gauss();
    ham_old += 0.5*(gp1[i]*gp1[i] + gp2[i]*gp2[i]);
  }
  
  /* Now create the field and calculate its contributions to the action (end of the 'misuse') */
  /* squnrm is the fermion part of the action : */
  /*   S = R^dagger * R  =  g_fermion^dag * D^{-1 dag} * D^{-1} * g_fermion = g_fermion Q^-1 g_fermion */

  /* PF1 det(1/(Q^2 + mu^2)) */

#pragma parallel for shared(g_X) 
  for(i=0; i<GRIDPOINTS; i++) {
    g_X[i].s1 = (gauss() + I*gauss())/sqrt(2); //Gaussian fields R
    g_X[i].s2 = (gauss() + I*gauss())/sqrt(2);
  }
  squnrm = square_norm(g_X);
  
  // step iv): g_fermion = \phi = K^dag * g_X = K^dag * \xi
  gam5D_wilson(g_fermion, g_X);
  /* assign_diff_mul(g_fermion, g_X, 0.+I*sqrt(g_musqr)); */
  ham_old += squnrm;

  // Add the part for the fermion fields

  // Do the molecular dynamic chain
  /* the simple LF scheme */

  /* the second order minimal norm multi-timescale integrator*/
  /* MN2_integrator(g_steps, 2, g_steps*g_stepsize, 0.2); */

  /* This is the recursive implementation */
  /* in can be found in rec_lf_integrator.c|h */
  n_steps[0] = 100;
  leapfrog(n_steps[0], tau/n_steps[0]);
  
  // Calculate the new action and hamiltonian
  ham = 0;
  s_g = 0;

#pragma parallel for shared(gp1, gp2) reduction(+: s_g, ham)
  for (i=0; i<GRIDPOINTS; i++) {
    s_g += S_G(i);
    ham += 0.5*(gp1[i]*gp1[i] + gp2[i]*gp2[i]);
  }
  /* Sum_ij [(g_fermion^*)_i (Q^-1)_ij (g_fermion)_j]  =  Sum_ij [(g_fermion^*)_i (g_X)_i] */
  ham += s_g;
  // add in the part for the fermion fields.
  cg(g_X, g_fermion, ITER_MAX, DELTACG, &gam5D_SQR_musqr_wilson);
  ham += scalar_prod_r(g_fermion, g_X);
  
  exphdiff = exp(ham_old-ham);
  acc = accept(exphdiff);
 
#pragma parallel for shared(gauge1_old, gauge2_old, gauge1, gauge2)
  for(i=0; i<GRIDPOINTS; i++) {
    gauge1_old[i]=gauge1[i];
    gauge2_old[i]=gauge2[i];
  }
 
  s_g_old = s_g;
  return(acc);
}

int accept(const double exphdiff)
{
  int acc=0, i;
  double r[1];

  // the acceptance step
  if(exphdiff>=1) {
    acc = 1; 
    R += 1;
  }
  else {
    ranlxd(r,1);
    if(r[0]<exphdiff) {
      acc = 1;
      R += 1;
    }
    else {
      // get the old values for phi, cause the configuration was not accepted
      for (i=0; i<GRIDPOINTS; i++)
	{
	  gauge1[i]=gauge1_old[i];
	  gauge2[i]=gauge2_old[i];
	};
      calculatelinkvars();
      s_g = s_g_old;
    }
  }
  return acc;
}


void add_windingN(int n){
  double r[2];
  ranlxd(r, 2);
  int pos = (int) X1*X2*r[0]; 
  int sign = 1;
  if(r[1]>0.5) sign = -1;
  int i,j;
  int x2  = pos/X1;
  int x1  = pos%X2;

  double phase[n+1][n+1];

  /* for(i=0;i<n+2;i++) for(j=0;j<n+2;j++) { */
  /*     pos = idx((i+x1-1+X1)%X1,(j+x2-1+X2)%X2,X1); */
  /*     //   printf("plaq i j = %d %d, %f \n ",i,j, (gauge1[pos] + gauge2[right1[pos]] - gauge1[right2[pos]] - gauge2[pos]  )); */
      
  /*   } */
  

  for(i=0;i<n+1;i++) for(j=0;j<n+1;j++) {
      phase[i][j] = 0;
    }


  for(i=0;i<n+1;i++) {
    phase[0][i] = i*M_PI/(2*n);
    phase[n][i] = 3*M_PI/2 - i*M_PI/(2*n);
    phase[i][0] = 2*M_PI - i*M_PI/(2*n);
    phase[i][n] = M_PI/2 + i*M_PI/(2*n);
  }
  phase[0][0] = 2*M_PI;

  int zero =0;
  
  for(i=0;i<n;i++) for(j=0;j<n;j++) {
      zero = 0;
      pos = idx((i+x1)%X1,(j+x2)%X2,X1);
      
      if(i==0&&j==0) zero=1.;

      if(j+1<n+1) gauge2[pos] += sign*(phase[i][j+1]-phase[i][j] - zero*2*M_PI);
      if(i+1<n+1) gauge1[pos] += sign*(phase[i+1][j]-phase[i][j]);
    }
  
  for(j=0;j<n;j++) {
    pos = idx((n+x1)%X1,(j+x2)%X2,X1);
    gauge2[pos] += sign*(phase[n][j+1]-phase[n][j]) ;
  }
  for(j=0;j<n;j++) {
    pos = idx((j+x1)%X1,(n+x2)%X2,X1);
    gauge1[pos] += sign*(phase[j+1][n]-phase[j][n]);
  }
  
  for(i=0;i<n+2;i++) for(j=0;j<n+2;j++) {
      pos = idx((i+x1-1+X1)%X1,(j+x2-1+X2)%X2,X1);
      //      printf("plaq i j = %d %d, %f \n ",i,j, (gauge1[pos] + gauge2[right1[pos]] - gauge1[right2[pos]] - gauge2[pos]  ));
           
    }



  /*
    for(i=0;i<6;i++){
      pos = idx((i+x1)%X1,(X2-1+x2)%X2,X1);      
      gauge2[pos] += sign*(phase[i][0]);
      pos = idx((i+x1)%X1,(X2+5+x2)%X2,X1);      
      gauge2[pos] += sign*(-phase[i][5]);
      pos = idx((X1-1+x1)%X1,(i+x2)%X2,X1);
      gauge1[pos] += sign*(phase[0][i]);
      pos = idx((X1+5+x1)%X1,(i+x2)%X2,X1);
      gauge1[pos] += sign*(-phase[5][i]);
      
    }
  */
  calculatelinkvars();

}











