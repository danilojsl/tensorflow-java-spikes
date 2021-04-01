import org.tensorflow.Tensor;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class TensorSpike {

    public static void main(String[] args) {

        float[] floats = {1f, 2f, 3f};

        Tensor<TFloat32> tfloats = TFloat32.vectorOf(floats);

        ByteBuffer bbuf = ByteBuffer.allocate(16).order(ByteOrder.nativeOrder());
        bbuf.clear();

        tfloats.rawData().copyTo(DataBuffers.of(bbuf), tfloats.numBytes());
        float resultFirstElement = bbuf.asFloatBuffer().get(0);
        System.out.println(resultFirstElement);

        FloatBuffer foreignBuffer = ByteBuffer.allocate((int) tfloats.numBytes())
                .order(ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN
                        ? ByteOrder.BIG_ENDIAN
                        : ByteOrder.LITTLE_ENDIAN).asFloatBuffer();

        tfloats.rawData().asFloats().copyTo(DataBuffers.of(foreignBuffer), foreignBuffer.capacity());
        float[] result = new float[foreignBuffer.remaining()];
        foreignBuffer.get(result);

        System.out.println(Arrays.toString(result));

    }

}
