import org.junit.Test;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.types.*;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class TensorTest {

    private static final double EPSILON = 1e-7;

    @Test
    /*public void readFromRawData() {
        int[] ints = {1, 2, 3};
        float[] floats = {1f, 2f, 3f};
        double[] doubles = {1d, 2d, 3d};
        long[] longs = {1L, 2L, 3L};
        boolean[] bools = {true, false, true};

        try (Tensor<TInt32> tints = TInt32.vectorOf(ints);
             Tensor<TFloat32> tfloats = TFloat32.vectorOf(floats);
             Tensor<TFloat64> tdoubles = TFloat64.vectorOf(doubles);
             Tensor<TInt64> tlongs = TInt64.vectorOf(longs);
             Tensor<TBool> tbools = TBool.vectorOf(bools)) {

            // validate that any datatype is readable with ByteBuffer (content, position)
            {
                ByteBuffer bbuf = ByteBuffer.allocate(1024).order(ByteOrder.nativeOrder());

                clearBuffer(bbuf); // FLOAT
                assertEquals(tfloats.numBytes(), tfloats.rawData().size());
                tfloats.rawData().copyTo(DataBuffers.of(bbuf), tfloats.numBytes());
                assertEquals(floats[0], bbuf.asFloatBuffer().get(0), EPSILON);

                clearBuffer(bbuf); // DOUBLE
                assertEquals(tdoubles.numBytes(), tdoubles.rawData().size());
                tdoubles.rawData().copyTo(DataBuffers.of(bbuf), tdoubles.numBytes());
                assertEquals(doubles[0], bbuf.asDoubleBuffer().get(0), EPSILON);

                clearBuffer(bbuf); // INT32
                assertEquals(tints.numBytes(), tints.rawData().size());
                tints.rawData().copyTo(DataBuffers.of(bbuf), tints.numBytes());
                assertEquals(ints[0], bbuf.asIntBuffer().get(0));

                clearBuffer(bbuf); // INT64
                assertEquals(tlongs.numBytes(), tlongs.rawData().size());
                tlongs.rawData().copyTo(DataBuffers.of(bbuf), tlongs.numBytes());
                assertEquals(longs[0], bbuf.asLongBuffer().get(0));

                clearBuffer(bbuf); // BOOL
                assertEquals(tbools.numBytes(), tbools.rawData().size());
                tbools.rawData().copyTo(DataBuffers.of(bbuf), tbools.numBytes());
                assertEquals(bools[0], bbuf.get(0) != 0);
            }

            // validate the use of direct buffers
            {
                ByteBuffer bbuf =
                        ByteBuffer.allocateDirect((int)tdoubles.numBytes()).order(ByteOrder.nativeOrder());
                tdoubles.rawData().copyTo(DataBuffers.of(bbuf), tdoubles.numBytes());
                assertEquals(doubles[0], bbuf.asDoubleBuffer().get(0), EPSILON);
            }

            // validate byte order conversion
            {
                DoubleBuffer foreignBuf =
                        ByteBuffer.allocate((int)tdoubles.numBytes())
                                .order(
                                        ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN
                                                ? ByteOrder.BIG_ENDIAN
                                                : ByteOrder.LITTLE_ENDIAN)
                                .asDoubleBuffer();
                tdoubles.rawData().asDoubles().copyTo(DataBuffers.of(foreignBuf), foreignBuf.capacity());
                double[] actual = new double[foreignBuf.remaining()];
                foreignBuf.get(actual);
                assertArrayEquals(doubles, actual, EPSILON);
            }
        }
    }*/

    private static void clearBuffer(Buffer buf) {
        buf.clear();
    }

}
