env_name="fcn"
env_dir="$HOME/envs"
apex_dir="$HOME/tools"
launch_file="launch_env.sh"

conda_modules="./fcn_env.yml"
pip_modules="./pip_vars.txt"

env_prefix="$env_dir/$env_name"
curr_dir=${PWD}

# check if directory for envs exists
if [ ! -d "$HOME/envs" ]; then
  mkdir $HOME/envs
fi

echo "building environment in $HOME/envs/fcn"
conda env create --prefix $env_prefix -f $conda_modules
source ~/.bashrc

echo "launching env and installing additional requirements"
conda activate $env_prefix
pip install -r $pip_modules

# check if apex exists
if [ ! -d "$apex_dir" ]; then
  mkdir -p $apex_dir
fi
cd $apex_dir

if [ ! -d "./apex" ]; then
  echo "downloading apex"
  git clone https://github.com/NVIDIA/apex
fi
cd ./apex

echo "installing apex [might take an hour...]"
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd $curr_dir
#remove old launch file if exists
if [ -f "$launch_file" ] ; then
  rm "$launch_file"
fi

echo "building launch file"
echo "module load cray" >> $launch_file
echo "module load cray-python" >> $launch_file
echo "source $HOME/.bashrc" >> $launch_file
echo "conda activate $env_prefix" >> $launch_file

