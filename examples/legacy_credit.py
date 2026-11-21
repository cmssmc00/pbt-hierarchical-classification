"""Historical CREDIT components on random tensors, never on patient records.

Run: python examples/legacy_credit.py
The main PBT CLI does not invoke this ensemble/student workflow.
"""

import tensorflow as tf

from pbt.legacy.ced import ced_loss, stack_teacher_logits
from pbt.legacy.models import build_credit_student, build_fusion_teacher


def main():
    tf.keras.utils.set_random_seed(7)
    architecture = dict(
        feature_dim=8, num_classes=10, projection_dim=8, attention_dim=4, fusion_dim=8, use_ihc=False, l2=0.0
    )
    teachers = [build_fusion_teacher(**architecture) for _ in range(2)]
    student = build_credit_student(**architecture)
    inputs = {"patches": tf.random.normal((2, 6, 8)), "patch_mask": tf.ones((2, 6), dtype=tf.bool)}
    teacher_logits = tf.stop_gradient(stack_teacher_logits(teachers, inputs))
    with tf.GradientTape() as tape:
        loss = ced_loss(student(inputs, training=True), teacher_logits, 2.5)
    gradients = tape.gradient(loss, student.trainable_variables)
    tf.debugging.assert_all_finite(loss, "Nonfinite synthetic CED loss")
    assert any(g is not None for g in gradients)
    print("Synthetic CREDIT forward/backward check passed; no clinical result is implied.")


if __name__ == "__main__":
    main()
