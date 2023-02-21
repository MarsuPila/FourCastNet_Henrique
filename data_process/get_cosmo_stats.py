#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import numpy as np
import h5py
import glob
from netCDF4 import Dataset as ncDataset

def get_fill_val(file_path, channels):
    dat = []
    with ncDataset(file_path, 'r') as _f:
      for var in channels:
        arr = _f[var][:].squeeze()
        if len(arr.shape)<3:
          arr = arr[None,:]
        dat.append(arr)
        if np.ma.is_masked(arr):
            return arr.get_fill_value()


def open_file(file_path, channels):
    dat = []
    with ncDataset(file_path, 'r') as _f:
      for var in channels:
        arr = _f[var][:].squeeze()
        if len(arr.shape)<3:
          arr = arr[None,:]
        if not np.ma.is_masked(arr):
            arr = np.ma.masked_array(arr, mask=np.zeros_like(arr, dtype=bool))
            if np.ma.min(arr) < -1.E35:
              print(var)
        dat.append(arr)
    dat = np.ma.concatenate(dat, axis=0)[None,:]
    return dat


if __name__ == "__main__":
    channels = ['U_10M', 'V_10M', 'PMSL', 'PS', 'T_2M', 'FI', 'T', 'U', 'u_100', 'V', 'v_100', 'RELHUM']
    years = [2015]
    data_dir = '/home/mkoch/Documents/projects/cscs/data/meteoswiss_extracted'
    batch_size = 48

    year_dir = data_dir + '/' + str(years[0])
    days = glob.glob(year_dir + '/*/', recursive=True)
    hrly = glob.glob(days[0] + '/*.nc', recursive=True)
    fill_value = get_fill_val(hrly[0], channels)

    means = np.zeros((1,42,1,1))
    stds  = np.zeros((1,42,1,1))

    # make tensor consisting of hourly data
    dat = []
    batches = 0
    total_steps = 0
    for year in years:
        year_dir = data_dir + '/' + str(year)
        days = glob.glob(year_dir + '/*/', recursive=True)
        for ii, day in enumerate(days):
            hrly = glob.glob(day + '/*.nc', recursive=True)
            for fl in hrly:
                dat.append(open_file(fl, channels))
                total_steps += 1
                if total_steps%batch_size == 0:    # split up data to prevent overflow
                    dat = np.ma.concatenate(dat, axis=0)
                    means += np.ma.mean(dat, keepdims=True, axis = (0,2,3))
                    stds  +=  np.ma.var(dat, keepdims=True, axis = (0,2,3))
                    batches += 1
                    dat   = []

    if len(dat)>0:
        batch_frac = float(len(dat))/batch_size
        dat    = np.ma.concatenate(dat, axis=0)
        means += batch_frac*np.ma.mean(dat, keepdims=True, axis = (0,2,3))
        stds  += batch_frac*np.ma.var(dat, keepdims=True, axis = (0,2,3))
        batches += batch_frac

    means = means/batches
    stds  = np.sqrt(stds/batches)

    # np.save(data_dir+'/global_means.npy', means)
    # np.save(data_dir+'/global_stds.npy',  stds)

    print("means: ", np.max(means))
    print("stds: ",  np.max(stds))
