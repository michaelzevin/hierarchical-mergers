#!/bin/bash/

method=NGNG
pop=fixed

python generate_mergers \
--method ${method} \
--firstgen-pop ${pop} \
--firstgen-samples /Users/michaelzevin/research/ligo/data_releases/populations/Population_Samples/default/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json \
--Ntree 10000 \
--Nhierarch 10 \
--sensitivity design_network \
--Vesc 30 50 100 200 300 500 1000 \
--BH-budget 50 100 500 1000 \
--verbose \
--output-path /Users/michaelzevin/research/hierarchical_mergers/data/${method}_${pop}_tree.hdf5 \
#--multiproc 1

