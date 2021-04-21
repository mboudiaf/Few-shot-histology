METHODS="simpleshot protonet maml finetune"
TEST_SOURCES="nct"
SHOTS="1 5"
for method in $METHODS
do
    for source in $TEST_SOURCES
    do
        for shot in ${SHOTS}
        do
            bash scripts/test.sh ${method} ${shot} crc-tp ${source}
        done
    done
done