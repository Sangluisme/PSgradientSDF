
#ifndef OPTIMIZER_SETTINGS_
#define OPTIMIZER_SETTINGS_

#include <nlohmann/json.hpp>

// parse loss function
enum LossFunction
{
    L2 = 0,
    CAUCHY = 1,
    HUBER = 2,
    TUKEY = 3,
    TRUNC_L2 = 4
};

// parse image formation model
enum ModelType {
    SH1,
    SH2,
    LED
}; 

struct OptimizerSettings {
    int max_it; // maximum number of iterations
    float conv_threshold; // threshold for convergence
    float damping; // LM damping term (often referred to as lambda)
    float lambda; // lambda for weight function (not for LM!)
    float lambda_sq;
    float reg_weight_rho;
    float reg_weight_n;
    float reg_weight_l;
    int order;
    bool upsample;
    ModelType model;
    LossFunction loss; // type of (robust) loss
    
    OptimizerSettings() :
        max_it(100),
        conv_threshold(1e-4),
        damping(1.0),
        lambda(0.5),
        lambda_sq(lambda * lambda),
        reg_weight_rho(0.0),
        reg_weight_n(0.0),
        reg_weight_l(0.0),
        order(1),
        upsample(false),
        model(SH1),
        loss(CAUCHY)
    {}

    OptimizerSettings(float reg_r, float reg_n, float reg_l) :
        max_it(100),
        conv_threshold(1e-4),
        damping(1.0),
        lambda(0.5),
        lambda_sq(lambda * lambda),
        reg_weight_rho(reg_r),
        reg_weight_n(reg_n),
        reg_weight_l(reg_l),
        order(1),
        upsample(false),
        model(SH1),
        loss(CAUCHY)
    {}

    OptimizerSettings(float reg_r, float reg_n, float reg_l, ModelType model_type, LossFunction loss_fun) :
        max_it(100),
        conv_threshold(1e-4),
        damping(1.0),
        lambda(0.5),
        lambda_sq(lambda * lambda),
        reg_weight_rho(reg_r),
        reg_weight_n(reg_n),
        reg_weight_l(reg_l),
        order(1),
        upsample(false),
        model(model_type),
        loss(loss_fun)
    {}
};
#endif // OPTIMIZER_SETTINGS_