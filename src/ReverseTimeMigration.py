"""
Reverse Time Migration

1.Load the seismic observed data and source Function
2.Forward modeling to generate the synthetic data   
3.Backward modeling 
"""
import SH_PSV_forward_modelng as fw
import SH_PSV_backward_modeling as bk
import cupy as cp
import numpy as np
import os 
import matplotlib.pyplot as plt
plt.style.use('fast')
"""
RTMの考え方 Claerbout(1971)
「地中の反射体は、下向き伝搬波の初到達時刻が上向き伝搬波の時刻と一致する地点に存在する」

1. import observed data and source function.
2. Forward modeling to generate the synthetic data.
3. Backward modeling from observed receriver data.
4. calculate the cross-correlation between the forward and backward modeling data.
5. Update the velocity model.

"""
class ReverseTimeMigration:
    """
    assume one source because of the v estimation
    """
    def __init__(self, **kwargs):
        # 1. import observing data and source function
        self.observed_u = kwargs['observed_u'] if 'observed_u' in kwargs else None        
        self.observed_v = kwargs['observed_v'] if 'observed_v' in kwargs else None
        self.observed_w = kwargs['observed_w'] if 'observed_w' in kwargs else None

        self.source_u = kwargs['source_u'] if 'source_u' in kwargs else None
        self.source_v = kwargs['source_v'] if 'source_v' in kwargs else None
        self.source_w = kwargs['source_w'] if 'source_w' in kwargs else None

        ## import location of the receiver
        self.receiver_loc = kwargs['receiver_loc'] if 'receiver_loc' in kwargs else None # np.array(Num_sensor, dtype=np.float32), 
        self.receiver_num = len(self.receiver_loc) if self.receiver_loc is not None else None

        ## import location of the source
        self.source_loc = kwargs['source_loc']  if 'source_loc' in kwargs else None # np.array(Num_sensor, dtype=np.float32)
        self.source_num = self.source_loc.size if self.source_loc is not None else None
        self.source_wavelet = kwargs['source_wavelet'] if 'source_wavelet' in kwargs else None
        self.fs = kwargs['fs'] if 'fs' in kwargs else None

        ## import the rho / estimate rho
        self.rho = kwargs['rho'] if 'rho' in kwargs else 1500
        self.poisson = kwargs['poisson_ratio'] if 'poisson_ratio' in kwargs else 0.33
        ## import sampling rate
        
        self.isnap = kwargs['isnap'] if 'isnap' in kwargs else 216
        self.nt = self.observed_u.shape[1]

        ## import other value for simulation
        self. absorbing_frame = kwargs['absorbing_frame'] if 'absorbing_frame' in kwargs else 50
        self.vmin = kwargs['vmin'] if 'vmin' in kwargs else 10.
        self.vmax = kwargs['vmax'] if 'vmax' in kwargs else 500.
        self.vstep = kwargs['vstep'] if 'vstep' in kwargs else 10
        self.v_fix = kwargs['v_fix'] if 'v_fix' in kwargs else None
        self.debug = kwargs['debug'] if 'debug' in kwargs else False
        self.receivers_height = kwargs['receivers_height'] if 'receivers_height' in kwargs else None 
        recerivers_height = self.receivers_height  - cp.max(self.receivers_height) if self.receivers_height is not None else None
        self.receivers_height = recerivers_height # recerivers_height is negative value. 0 is the highest point

        self.check_parameters()
        pass

    def check_parameters(self):
        if self.observed_u.shape != self.observed_v.shape or self.observed_u.shape != self.observed_w.shape:
            raise ValueError('The shape of observed data is not correct.')
        if self.source_u.shape != self.source_v.shape or self.source_u.shape != self.source_w.shape:
            raise ValueError('The shape of source function is not correct.')
        if self.receiver_loc is None:
            raise ValueError('The receiver location is not defined.set attribute receiver.')
        if self.source_loc is None:
            raise ValueError('The source location is not defined. set attribute source.')
        if self.fs is None:
            raise ValueError('The sampling rate is not defined.') 
        if self.receivers_height is not None:
            if self.receivers_height.size != self.receiver_num:
                raise ValueError('The size of the receiver height is not correct.')   
        pass

    def estimate_velocity(self, array:cp.array, vmin, vmax, method='closs_corriation', vstep = 10):
        print('Estimate the velocity')
        M, L = array.shape
        r = (vmax/vmin)**(1/(vstep-1))
        v_array = vmin*r**cp.arange(vstep)
        sum_max = 0
        v_estimated = vmin
        if method == 'closs_corriation':
            for v in v_array:
                abs_d = cp.abs(self.receiver_loc - self.source_loc)
                tsteps = (self.fs * abs_d / v)# array(Num_sensor, dtype=np.float32)
                tsteps = tsteps.astype(cp.int32)
                tsteps -= cp.min(tsteps) # tstep offset of each sensor ch
                L_t = L - cp.max(tsteps)
                # インデックス配列を作成
                indices = tsteps[:, None] + cp.arange(L_t)
                # シフトされた配列を取得
                shifted = array[cp.arange(array.shape[0])[:, None], indices]
                products = cp.prod(shifted, axis=0)
                sum_v = cp.sum(products)
                sum_v = cp.absolute(sum_v)
                # find the maximum value
                if sum_v > sum_max:
                    sum_max = sum_v
                    v_estimated = v
                continue
        else:
            raise ValueError('The method is not implemented.')
        print(f'\nVelocity Estimation is finished.\nThe estimated velocity is {cp.round(v_estimated, 2)}m/s')
        return v_estimated
    
    def __set_conditions_for_fwbw_modeling(self, estimated_v, CFL, absorbing_frame):
        dx = estimated_v / self.fs / CFL
        dz = dx
        nx = int((cp.max(self.receiver_loc) - cp.min(self.receiver_loc)) / dx)
        nx += 4*absorbing_frame
        nz = nx
        rho = self.rho * cp.ones((nx, nz), dtype=cp.float32)
        vs = estimated_v * cp.ones((nx, nz), dtype=cp.float32)
        vp = cp.sqrt((2 * self.poisson + 1) / (1 - 2 * self.poisson)) * vs
        return dx, dz, nx, nz, rho, vs, vp
    
    def __set_isnap_for_allocated_memory(self, total_memory, memory_merge, nx, nz, nt):
        """
        assume Dtype = float32, dtype_size = 8
        """
        dtype_size = 8
        allowed_memory = total_memory - memory_merge # MiB
        max_steps = allowed_memory * 1024 * 1024 // (nx * nz * dtype_size) // 3
        if max_steps == 0:
            isnap = nt # if the memory is not enough, the isnap is set to nt
        else:
            isnap = int(cp.ceil(nt / max_steps).item())
        return isnap

    def run(self, total_memory = 24000, memory_merge = 2000, backwardmodel_method = 'closs_correlation'):
        """
        run the Reverse Time Migration
        1. estimate the velocity
        2. Forward and backward modeling to generate the synthetic data
        3. calculate the cross-correlation between the forward and backward modeling data

        Parameters
        ----------------------------------------------
        total_memory : int
            the total memory for the simulation (MiB)
        memory_merge : int
            the memory for the other process (MiB)
        ----------------------------------------------

        backwardmodel_method : str
        'closs_correlation' : calculate the cross-correlation between the forward and backward modeling data
        'convolution' : calculate the convolution between the forward and backward modeling data
        """    
        # estimate velocity
        if self.v_fix is not None:
            print(f'the velocity is selected as fixed {self.v_fix}m/s')
            estimated_v = self.v_fix
        else:
            estimated_v = self.estimate_velocity(self.observed_v, self.vmin, self.vmax, vstep = self.vstep)

        # 2. Forward and backward modeling to generate the synthetic data
        CFL = 0.8
        #isnap = self.isnap
        absorbing_frame = self.absorbing_frame

        flag = 99 # is simulation still running ?? -> 1: yes, 0: no
        print('\nstart forward modeling')
        while flag:
            CFL *= 0.5 # decrease the CFL to avoid the error
            flag = 0 # reset the flag

            dx, dz, nx, nz, rho, vs, vp = self.__set_conditions_for_fwbw_modeling(estimated_v, CFL, absorbing_frame)
            fs = self.fs
            nt = self.nt
            receiver_loc = self.receiver_loc
            wavelet_u = self.source_u
            wavelet_v = self.source_v
            wavelet_w = self.source_w
            observed_u = self.observed_u
            observed_v = self.observed_v
            observed_w = self.observed_w
            receivers_height = self.receivers_height

            receiver_loc_step = []
            offset = 2*absorbing_frame
            for loc in self.receiver_loc: 
                receiver_loc_step.append([int(loc/dx+offset), 1])
            src_loc_step = [[int(self.source_loc/dx+offset), 1]]
            surface_matrix = cp.ones((nx, nz), dtype=cp.float32)
            height_array = cp.zeros((nx), dtype=cp.float32)
            steepness_array = cp.zeros_like(height_array) # 0 steepness in the boundary
            if receivers_height is not None:
                # get receiver_height_steps
                receivers_height_step = -1 * cp.ones_like(receivers_height) * receivers_height / dz
                
                for i,locstep in enumerate(receiver_loc_step):
                    heightstep = receivers_height_step[i]
                    receiver_loc_step[i][1] = int(heightstep)
                    # side boundary condition
                    if i == 0:
                        surface_matrix[:locstep[0],:heightstep] = 0
                        height_array[:locstep[0]] = heightstep

                    if i > 0:
                        next_locstep = receiver_loc_step[i][0]
                        locstep = receiver_loc_step[i-1][0]
                        next_heightstep = receivers_height_step[i]
                        heightstep = receivers_height_step[i-1]
                        for ix in range(locstep, next_locstep):
                            lean = (next_heightstep - heightstep) / (next_locstep - locstep)
                            width = (ix - locstep)
                            height_ix = lean * width + heightstep
                            surface_matrix[ix,:int(height_ix)] = 0
                            height_array[ix] = height_ix
                
                height_array[receiver_loc_step[-1][0]:] = receivers_height_step[-1]
                surface_matrix[receiver_loc_step[-1][0]:, :int(receivers_height_step[-1])] = 0

                self.height_array = height_array

                ## make steepness array
                for i in range(1, len(height_array)-1):
                    steepness_array[i] = (height_array[i+1] - height_array[i-1])*dz / dx
                self.steepness_array = steepness_array

                if self.debug:
                    plt.plot(height_array.get())
                    plt.title('Surface Shape')
                    plt.show()

                    plt.imshow(surface_matrix.get().T)
                    plt.title('Surface Matrix')
                    plt.show()

                self.surface_matrix = surface_matrix
                # set source location_steps
                src_loc_ind = src_loc_step[0][0]
                source_height_step = height_array[src_loc_ind]
                src_loc_step[0][1] = int(source_height_step + 1 )# source is upper than the receiver

            self.src_loc_step = src_loc_step
            # calcurate memory for
            isnap = self.isnap if self.debug else self.__set_isnap_for_allocated_memory(total_memory, memory_merge, nx, nz, nt)
            # forward modeling
            _fw = fw.forward_modeling(nx = nx,
                                       nz = nz,
                                       dx = dx,
                                       dz = dz,
                                       nt = nt,
                                       fs = fs,
                                       vs = vs,
                                       vp = vp,
                                       rho = rho,
                                       absorbing_frame = absorbing_frame,
                                       src_loc = src_loc_step,   
                                       wavelet_u = wavelet_u,
                                       wavelet_v = wavelet_v,
                                       wavelet_w = wavelet_w,
                                       receiver_loc = receiver_loc_step,
                                       isnap = isnap,
                                       receivers_height = receivers_height,
                                       surface_matrix = surface_matrix,
                                       steepness_array = steepness_array
                                       )
            flag = _fw.run(show=self.debug,save=True) # if error, run returns 1 or 2 ro 3

            if flag:
                continue
            else:
                print('The forward modeling is safe. pass to the backward modeling.')
                pass
            # backward modeling
            _bw = bk.backward_modeling(nx = nx,
                                        nz = nz,
                                        dx = dx,
                                        dz = dz,
                                        nt = nt,
                                        fs = fs,
                                        vs = vs,
                                        vp = vp,
                                        rho = rho,
                                        absorbing_frame = absorbing_frame,
                                        src_loc = src_loc_step,
                                        observed_data_u = observed_u,
                                        observed_data_v = observed_v,
                                        observed_data_w = observed_w,
                                        receiver_loc = receiver_loc_step,
                                        isnap = _fw.isnaps,
                                        receivers_height = receivers_height,
                                        surface_matrix = surface_matrix,
                                        steepness_array = steepness_array
                                        ) # isnap is 
            flag = _bw.run_calc(show = self.debug,
                                import_fwdata_u = _fw.u_save,
                                import_fwdata_v = _fw.v_save,
                                import_fwdata_w = _fw.w_save,
                                isnaps = _fw.isnaps,
                                save = False,
                                method =  backwardmodel_method,
                                )

            if flag:
                continue
            else:
                print('\nThe backward modeling is safe. calcurated the cross-correlation.')
                pass

        print(f'\nThe simulation is finished.')
        print(f'The estimated velocity = {estimated_v}m/s')
        print(f'The CFL = {CFL}')
        print(f'nx = {nx}, nz = {nz}')
        print(f'dx = {dx}')

        self.image_u = _bw.result_u
        self.image_v = _bw.result_v
        self.image_w = _bw.result_w

        self.dx = dx
        self.dz = dz
        self.nx = nx
        self.nz = nz
        self.offset = offset * dx
        self.CFL = CFL
    
    def save_result(self, dir, savename):
        # 軸の範囲を定義
        xmin = -self.offset
        zmin = 0
        xmax = self.dx * self.nx - self.offset
        zmax = self.dz * self.nz
        xmin = float(xmin)
        xmax = float(xmax)
        zmin = float(zmin)
        zmax = float(zmax)
        surface_matrix = self.surface_matrix if hasattr(self, 'surface_matrix') else None
        import os
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.savez_compressed(dir + savename,
                 u = self.image_u,
                 v = self.image_v,
                 w = self.image_w,
                 receiver_loc = self.receiver_loc,
                 src_loc_step = self.src_loc_step,
                 offset = self.offset,
                 xmin = xmin,
                 xmax = xmax,
                 zmin = zmin,
                 zmax = zmax,
                 dx = self.dx,
                 dz = self.dz,
                 nx = self.nx,
                 nz = self.nz,
                 surface_matrix = surface_matrix
                 )

    def show_result(self, save = False, dir = None, savename = None, cmap='gray', mean = False):

        # attenuate values at src_loc_step for imaging
        src_loc_step = self.src_loc_step
        self.image_u[src_loc_step[0][0] -10:src_loc_step[0][0] +10, src_loc_step[0][1]: src_loc_step[0][1]+10] *= 1e-3
        self.image_v[:, src_loc_step[0][1]: src_loc_step[0][1]+10] *= 1e-1
        self.image_w[src_loc_step[0][0] -10:src_loc_step[0][0] +10, src_loc_step[0][1]: src_loc_step[0][1]+10] *= 1e-3
        
        # 軸の範囲を定義
        xmin = -self.offset
        zmin = 0
        xmax = self.dx * self.nx - self.offset
        zmax = self.dz * self.nz
        xmin = float(xmin)
        xmax = float(xmax)
        zmin = float(zmin)
        zmax = float(zmax)

        umap = self.image_u.get().T
        vmap = self.image_v.get().T
        wmap = self.image_w.get().T
        if mean:
            umap = umap - cp.mean(umap)
            vmap = vmap - cp.mean(vmap)
            wmap = wmap - cp.mean(wmap)
        
        umax = cp.max(umap) if cp.max(umap) > -cp.min(umap) else -cp.min(umap)
        vmax = cp.max(vmap) if cp.max(vmap) > -cp.min(vmap) else -cp.min(vmap)
        wmax = cp.max(wmap) if cp.max(wmap) > -cp.min(wmap) else -cp.min(wmap)

        # 図とサブプロットを作成
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 各サブプロットに画像を表示
        im1 = axes[0].imshow(umap, cmap=cmap, extent=[xmin, xmax, zmax, zmin], vmax = umax, vmin = -umax)

        axes[0].set_title('Reverse Time Migration, u')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('z')

        im2 = axes[1].imshow(vmap, cmap=cmap, extent=[xmin, xmax, zmax, zmin], vmax = vmax, vmin = -vmax)
        axes[1].set_title('Reverse Time Migration, v')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('z')

        im3 = axes[2].imshow(wmap, cmap=cmap, extent=[xmin, xmax, zmax, zmin], vmax = wmax, vmin = -wmax)
        axes[2].set_title('Reverse Time Migration, w')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('z')

        # レイアウトを調整して表示
        plt.tight_layout()
        plt.ioff()  # インタラクティブモードをオフにする
        if save:
            if dir is None or savename is None:
                raise ValueError('The directory and savename are not defined.')
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(dir + savename + '.png')
        else:
            plt.show()

if __name__ == '__main__':

    hr = wa.Hr2c('example/seismogram_ex7.npz')
    time_to = 0.3
    hr.timepick('y', 0, time_to)
    hr.timepick('x', 0, time_to)
    hr.timepick('z', 0, time_to)
    hr.show('y')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(np.load('example/seismogram_ex7.npz')['rho'].T, cmap='gray')
    plt.ion()
    plt.show()
    # 1. import observing data and source function
    fs = cp.float32(hr.fs)
    observed_u = cp.array(hr.x)
    observed_v = cp.array(hr.y)
    observed_w = cp.array(hr.z)
    source_ch = cp.int32(hr.get_source_ch())
    source_u = cp.array(hr.x[source_ch][0:int(hr.fs*0.02)])
    source_v = cp.array(hr.y[source_ch][0:int(hr.fs*0.02)])
    source_w = cp.array(hr.z[source_ch][0:int(hr.fs*0.02)])

    receiver_loc = cp.array(hr.distance)
    source_loc = cp.array(hr.source_x)
    absorbing_frame = 100
    isnap = 1

    RTM = ReverseTimeMigration(observed_u = observed_u,
                                observed_v = observed_v,
                                observed_w = observed_w,
                                source_u = source_u,
                                source_v = source_v,
                                source_w = source_w,
                                receiver_loc = receiver_loc,
                                source_loc = source_loc,
                                absorbing_frame = absorbing_frame,
                                isnap = isnap,
                                fs = fs,
                                vmin = 80,
                                vmax = 300,
                                vstep = 2000,
                                v_fix = 200,
                                debug = False,
                                )
    RTM.run(total_memory = 24000, memory_merge = 4000)
    RTM.show_result(cmap = 'seismic', mean = True)