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


int R;
int g_cgiterations1;
int g_cgiterations2;

int update() //Basic HMC update step
{
  int i, acc;
  double exphdiff;
  
  /* the new impulses and the 'generator' of the arbitrary pseudofield */
  /* calculate the hamiltonian of this state: new impulses + action */
  /* g_X is ab-used a bit - here it is \xi = (gamma5 D)^{-1} \phi */
  
  ham_old = s_g_old;
  for(i=0; i<GRIDPOINTS; i++) {
    gp1[i] = gauss();
    gp2[i] = gauss();
    ham_old += 0.5*(gp1[i]*gp1[i] + gp2[i]*gp2[i]);
  }
  
  /* Now create the field and calculate its contributions to the action (end of the 'misuse') */
  /* squnrm is the fermion part of the action : */
  /*   S = R^dagger * R  =  g_fermion^dag * D^{-1 dag} * D^{-1} * g_fermion = g_fermion Q^-1 g_fermion */

  
  n_steps[0] = 100;
  leapfrog(n_steps[0], tau/n_steps[0]);
  
  // Calculate the new action and hamiltonian
  ham = 0;
  s_g = 0;
  for (i=0; i<GRIDPOINTS; i++) {
    s_g += S_G(i);
    ham += 0.5*(gp1[i]*gp1[i] + gp2[i]*gp2[i]);
  }
  /* Sum_ij [(g_fermion^*)_i (Q^-1)_ij (g_fermion)_j]  =  Sum_ij [(g_fermion^*)_i (g_X)_i] */
  ham += s_g;
  
  exphdiff = exp(ham_old-ham);
  acc = accept(exphdiff);
 
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







