#ifndef BOOSTING_STATIC_CHECKS_
#define BOOSTING_STATIC_CHECKS_

#define boosting_static_data_type_check(condition, message) static_assert( !(condition), #message )

#endif //BOOSTING_STATIC_CHECKS_
