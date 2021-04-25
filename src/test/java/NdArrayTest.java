import org.junit.Test;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.tensorflow.ndarray.NdArrays.vectorOfObjects;

public abstract class NdArrayTest<T> {

    protected abstract NdArray<T> allocate(Shape shape);

    protected abstract T valueOf(Long val);

    @Test
    public void iterateElements() {
        NdArray<T> matrix3d = allocate(Shape.of(5, 4, 5));

        matrix3d.scalars().forEachIndexed((coords, scalar) -> {
            scalar.setObject(valueOf(coords[2]));
        });

        assertEquals(valueOf(0L), matrix3d.getObject(0, 0, 0));
        assertEquals(valueOf(1L), matrix3d.getObject(0, 0, 1));
        assertEquals(valueOf(4L), matrix3d.getObject(0, 0, 4));
        assertEquals(valueOf(2L), matrix3d.getObject(0, 1, 2));

        matrix3d.elements(1).forEach(vector -> {
            vector.set(vectorOfObjects(valueOf(5L), valueOf(6L), valueOf(7L), valueOf(8L), valueOf(9L)));
        });

        assertEquals(valueOf(5L), matrix3d.getObject(0, 0, 0));
        assertEquals(valueOf(6L), matrix3d.getObject(0, 0, 1));
        assertEquals(valueOf(9L), matrix3d.getObject(0, 0, 4));
        assertEquals(valueOf(7L), matrix3d.getObject(0, 1, 2));

        long value = 0L;
        for (NdArray<T> matrix : matrix3d.elements(0)) {
            assertEquals(2L, matrix.shape().numDimensions());
            assertEquals(4L, matrix.shape().size(0));
            assertEquals(5L, matrix.shape().size(1));

            for (NdArray<T> vector : matrix.elements(0)) {
                assertEquals(1L, vector.shape().numDimensions()) ;
                assertEquals(5L, vector.shape().size(0));

                for (NdArray<T> scalar : vector.scalars()) {
                    assertEquals(0L, scalar.shape().numDimensions()) ;
                    scalar.setObject(valueOf(value++));
                    try {
                        scalar.elements(0);
                        fail();
                    } catch (IllegalArgumentException e) {
                        // as expected
                    }
                }
            }
        }
        assertEquals(valueOf(0L), matrix3d.getObject(0, 0, 0));
        assertEquals(valueOf(5L), matrix3d.getObject(0, 1, 0));
        assertEquals(valueOf(9L), matrix3d.getObject(0, 1, 4));
        assertEquals(valueOf(20L), matrix3d.getObject(1, 0, 0));
        assertEquals(valueOf(25L), matrix3d.getObject(1, 1, 0));
        assertEquals(valueOf(99L), matrix3d.getObject(4, 3, 4));
    }

}
