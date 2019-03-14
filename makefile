CXX=g++-4.9
CXXFLAGS=-c -std=c++11 -L/usr/local/lib -O3 -march=native -msse2 -I /usr/local/armadillo-6.700.7/include

all:collaborative_classification.o


#Project linking
collaborative_classification.o: types.o main.o regression_stump.o gentle_boost_ensemble.o bagging_utils.o bagged_boosting_ensemble.o
	$(CXX)  main.o types.o regression_stump.o gentle_boost_ensemble.o bagging_utils.o bagged_boosting_ensemble.o -o collaborative_classification.o  -lopenblas -llapack -lpthread

collaborative_classification_no_validation.o: types.o main_no_validation.o regression_stump.o gentle_boost_ensemble.o bagging_utils.o bagged_boosting_ensemble.o
	$(CXX)  main_no_validation.o types.o regression_stump.o gentle_boost_ensemble.o bagging_utils.o bagged_boosting_ensemble.o -o collaborative_classification_no_validation.o  -lopenblas -llapack -lpthread

gentle_boost.o: types.o multiprecision_types.hpp cpp_dec_float_50_support_eigen.hpp  regression_stump.o gentle_boost_ensemble.o main_gb.o
	$(CXX)  main_gb.o regression_stump.o gentle_boost_ensemble.o -o gentle_boost.o -lopenblas -llapack -lpthread

types.o: types.hpp types.cpp
	$(CXX) $(CXXFLAGS) types.cpp

regression_stump.o: regression_stump.hpp regression_stump.cpp gentle_boost_ensemble.hpp types.hpp
	$(CXX) $(CXXFLAGS) regression_stump.cpp -lopenblas -llapack -lpthread

gentle_boost_ensemble.o: gentle_boost_ensemble.hpp gentle_boost_ensemble.cpp multiprecision_types.hpp regression_stump.hpp types.hpp
	$(CXX) $(CXXFLAGS) gentle_boost_ensemble.cpp  -lopenblas -llapack -lpthread

bagging_utils.o: bagging_utils.hpp bagging_utils.cpp types.hpp
	$(CXX) $(CXXFLAGS) bagging_utils.cpp -lopenblas -llapack

bagged_boosting_ensemble.o: bagged_boosting_ensemble.hpp bagged_boosting_ensemble.cpp bagging_utils.hpp multiprecision_types.hpp gentle_boost_ensemble.hpp types.hpp
	$(CXX) $(CXXFLAGS) bagged_boosting_ensemble.cpp  -lopenblas -llapack -lpthread

main.o: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp

main_gb.o: main_gb.cpp
	$(CXX) $(CXXFLAGS) main_gb.cpp

main_no_validation.o: main_no_validation.cpp
	$(CXX) $(CXXFLAGS) main_no_validation.cpp
