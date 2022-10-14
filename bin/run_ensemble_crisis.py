from crisis_ensemble_exp import train_base_models, evaluate_base_models, evaluate_ensemble_models
import tensorflow as tf

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    train_base_models()
    evaluate_base_models()
    evaluate_ensemble_models()
    