data_path="/ssd/dataset/histology/converted"
all_sources="nct lc25000 kaggle_crc crc-tp bach breakhis"

for source in ${all_sources}
do
    source_path=${data_path}/${source}
    find ${source_path} -name '*.tfrecords' -type f -exec sh -c 'python3 -m tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${source_path} {} \;
done