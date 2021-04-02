package util;

import org.junit.Before;
import org.junit.Test;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;


public class TensorValuesTest {

    private static final float EPSILON_F = 1e-7f;
    Ops tf;
    Scope scope;

    @Before
    public void setUp() {
        EagerSession session = EagerSession.create();
        tf = Ops.create(session);
        scope = new Scope(session);
    }

    @Test
    public void shouldGet1DimensionPrimitiveArrayValuesWhen1DimensionTensorIsSent() {
        float[] expectedRawInput = {0.1f, 0.2f, 0.3f, 0.4f};
        Tensor<TFloat32> tensorInput = tf.constant(expectedRawInput).asTensor();

        float[] actualRawInput = TensorValues.castToPrimitiveFloat(tensorInput);

        assertArrayEquals(expectedRawInput, actualRawInput, EPSILON_F);
    }

    @Test
    public void shouldGet1DimensionPrimitiveArrayValuesWhen2DimensionTensorIsSent() {
        float[][] rawInput = {{0.1f, 0.2f}, {0.3f, 0.4f}};
        Tensor<TFloat32> tensorInput = tf.constant(rawInput).asTensor();
        float[] expectedRawOutput = {0.1f, 0.2f, 0.3f, 0.4f};

        float[] actualRawOutput = TensorValues.castToPrimitiveFloat(tensorInput);

        assertArrayEquals(expectedRawOutput, actualRawOutput, EPSILON_F);
    }

    //TODO: Check NdArray feature for Ndimensional operations

    @Test
    public void shouldCreateOneDimensionTensorOfRandomValuesWhenShapeIsSent() {
        int[] rawShape = {4};
        Operand<TInt32> shape = Constant.vectorOf(scope, rawShape);

        Tensor<TFloat32> tensorValues = TensorValues.initializeTruncatedNormalTensor(shape, scope);

        assertEquals(1, tensorValues.shape().numDimensions());
        assertEquals(4, tensorValues.shape().size(0));
    }

    @Test
    public void shouldCreateTwoDimensionTensorOfRandomValuesWhenShapeIsSent() {
        int[] shape = {3, 4};
        Operand<TInt32> dimensions = Constant.vectorOf(scope, shape);

        Tensor<TFloat32> tensorValues = TensorValues.initializeTruncatedNormalTensor(dimensions, scope);

        assertEquals(2, tensorValues.shape().numDimensions());
        assertEquals(3, tensorValues.shape().size(0));
        assertEquals(4, tensorValues.shape().size(1));
    }

}