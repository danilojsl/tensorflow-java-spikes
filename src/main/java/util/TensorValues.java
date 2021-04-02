package util;

import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Scope;
import org.tensorflow.op.random.TruncatedNormal;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class TensorValues {

    public static float[] castToPrimitiveFloat(Tensor<TFloat32> tensor) {
        FloatBuffer foreignBuffer = ByteBuffer.allocate((int) tensor.numBytes())
                .order(ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN
                        ? ByteOrder.BIG_ENDIAN
                        : ByteOrder.LITTLE_ENDIAN).asFloatBuffer();

        tensor.rawData().asFloats().copyTo(DataBuffers.of(foreignBuffer), foreignBuffer.capacity());
        float[] rawValues = new float[foreignBuffer.remaining()];
        foreignBuffer.get(rawValues);
        foreignBuffer.clear();

        return rawValues;
    }

    public static Tensor<TFloat32> initializeTruncatedNormalTensor(Operand<TInt32> shape, Scope scope) {
        TruncatedNormal.seed(1000L);
        TruncatedNormal<TFloat32> truncatedNormal = TruncatedNormal.create(scope, shape, TFloat32.DTYPE);

        return truncatedNormal.asTensor();
    }

}
