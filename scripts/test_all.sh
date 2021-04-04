METHODS="tim"
for method in $METHODS
do
    bash scripts/test.sh ${method} 1 breakhis crc-tp
    bash scripts/test.sh ${method} 5 breakhis crc-tp

    # bash scripts/test.sh ${method} 1 breakhis lc25000
    # bash scripts/test.sh ${method} 5 breakhis lc25000
done