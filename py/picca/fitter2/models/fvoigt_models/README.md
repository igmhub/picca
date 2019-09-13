# Fvoigt for HCD modelling

## Adding files in fvoigt_models :
  
**always** : Fvoigt_whatever.txt  :smile:

## How to use fvoigt_models : 

Modification in the file **.ini :

* In [model] use model-pk = *pk_hcd*

* Add a new section in the data.ini files : [hcd_model]

* Add in this section : name_hcd_model = *whatever*

* In [parameters] : add HCD parameters (see below)

## HCD parameters :

* *bias_hcd* = real bias hcd (measured in the DLA-autocorrelation) * $Fvoigt^{non-norm}(0)$
* *beta_hcd* = real beta hcd
* *L0_hcd* == 1.0 (scale factor) --> hard to fit due to degenerancy (has to be one)

## Description of each Fvoigt function in fvoigt_models directory

* exp : implementation in eBOSS DR14

* london_6.0 : Implementation for mocks london_6.0

* saclay_4.4: Implementation for mocks saclay_4.4

I show in the 06/05/19 DESI Lyman-alpha meeting that my implementation is the correct way to model the DLAs in Lyman-alpha autocorrelation function.

* DR12_noterdame : built with Noterdame/Pasquier DLAs catalogue and DR12 QSO catalogue

* DR12_prochaska : built with Prochaska DLAs catalogue and DR12 QSO catalogue
