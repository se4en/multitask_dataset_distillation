wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz
mkdir -p data && mkdir -p data/orig
tar xvf scicite.tar.gz -C data/orig
rm scicite.tar.gz
