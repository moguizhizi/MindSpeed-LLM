
## 1.镜像下载（待补充昇腾社区的镜像下载地址）
通过uname -a确认自身系统是ubuntu_x86 或者 openeuler
根据需要下载对应的镜像,如下为下载链接：
https://www.hiascend.com/developer/ascendhub/detail/7acc722a0e494eca9c90c028cb299275

## 2.镜像加载
```bash
# 挂载镜像
docker load -i modellink_*.tar
# 确认modellink:tag是否挂载成功                                         
docker image list    
```

## 3.创建镜像容器
注意当前默认配置驱动和固件安装在/usr/local/Ascend，如有差异请修改指令路径。
当前容器默认初始化npu驱动和CANN环境信息，如需要安装新的，请自行替换或手动source，详见容器的bashrc
```bash
# 挂载镜像
docker run -dit --ipc=host --network host --name 'modellink_test' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware  -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ modellink:tag
```

## 4.登录镜像并确认环境状态
```bash
# 登录容器
docker exec -it modellink_test /bin/bash                           
# 确认npu是否可以正常使用，否则返回3.检查配置
npu-smi info
```

## 5.拉取modellink
当前镜像推荐配套版本,用户可根据自己所需的版本配套，进行ModelLink和MindSpeed的更新使用。
```bash
# 从Gitee克隆ModelLink仓库 (git checkout master)
git clone https://gitee.com/ascend/ModelLink.git
cd ModelLink
git checkout 85f9126
# 从Gitee克隆MindSpeed仓库(git checkout f3332571)
git clone https://gitee.com/ascend/MindSpeed
cd MindSpeed
git checkout f3332571
```

## 6.拉取megatron并切换对应版本放到modellink下
```bash
# 登录容器
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.7.0
cp -r megatron ../ModelLink/
```

## 7.单机以及多机模型的预训练任务运行
基于拉取的镜像和仓库代码，执行单机和多机的预训练任务，具体可参考[ModelLink 使用指南](./USER_GUIDE.md)执行