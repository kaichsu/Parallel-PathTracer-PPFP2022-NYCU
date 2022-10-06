#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

// decrease exponents by 1 until 0 and return multmask has any 1 or not;
bool convert_mult_mask(__pp_vec_int &exponents, __pp_mask &dest_mult_mask, __pp_mask &valid_pass){

  __pp_vec_int vec_zero;
  _pp_vset_int(vec_zero, 0, valid_pass);

  __pp_mask vec_vgt_zero;
  _pp_vgt_int(vec_vgt_zero, exponents, vec_zero, valid_pass);

  __pp_vec_int vec_one;
  _pp_vset_int(vec_one, 1, valid_pass);
  _pp_vsub_int(exponents, exponents, vec_one, vec_vgt_zero);
  
  dest_mult_mask = vec_vgt_zero;
  return (_pp_cntbits(vec_vgt_zero) > 0)? 1: 0;
}

void clamp_float(__pp_vec_float &vec_source, __pp_mask &valid_pass, float limit = 9.999999){
  __pp_vec_float vec_limit = _pp_vset_float(limit);
  __pp_mask limit_mask = _pp_init_ones();

  _pp_vgt_float(limit_mask, vec_source, vec_limit, valid_pass);
  _pp_vmove_float(vec_source, vec_limit, limit_mask);
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    __pp_mask valid_pass = _pp_init_ones();
    //todo if i + vector width > = N, adjust allpass.
    if(i + VECTOR_WIDTH > N){
      valid_pass = _pp_init_ones(N - i);    
    }
    __pp_vec_float vec_value;
    __pp_vec_int vec_exponents;

    _pp_vload_float(vec_value, &values[i], valid_pass);
    _pp_vload_int(vec_exponents, &exponents[i], valid_pass);

    __pp_vec_float vec_result = _pp_vset_float(1.0);

    __pp_mask multMask;
    while(convert_mult_mask(vec_exponents, multMask, valid_pass)){
      _pp_vmult_float(vec_result, vec_result, vec_value, multMask);
      clamp_float(vec_result, valid_pass);
      // printf("test\n");
      // for (int i = 0; i < N; i++)
      //   printf("% f ", output[i]);
      // printf("\n");
    }
    _pp_vstore_float(&output[i], vec_result, valid_pass);

  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
  }

  return 0.0;
}