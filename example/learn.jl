using TensorQEC.Yao
reg = zero_state(2);

apply!(reg, put(2, 2=>X));

measure!(put(2,1=>X),reg)

