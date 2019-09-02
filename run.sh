#!/bin/bash

#source /opt/meituan/hadoop-gpu/bin/hadoop_user_login_centos7.sh hadoop-dpsr
source /opt/meituan/hadoop/bin/hadoop_user_login.sh hadoop-dpsr
source /opt/meituan/tensorflow-release/local_env.sh
export JAVA_HOME="/usr/local/java"
export HADOOP_HOME=/opt/meituan/hadoop
unset CLASSPATH
${AFO_TF_HOME}/bin/tensorflow-submit.sh -conf config.xml -files predict.py,model2,model2.vectors.npy
