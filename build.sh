cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=1 ..
make install -j4
pip install -e /home/zhaoguodong/work/code/MAPPO-MPC-OmniDrones/acados/interfaces/acados_template