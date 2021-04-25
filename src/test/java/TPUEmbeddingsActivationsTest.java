import org.tensorflow.EagerSession;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;

public class TPUEmbeddingsActivationsTest {

    EagerSession session = EagerSession.create();
    Ops tf = Ops.create(session);
    Scope scope = new Scope(session);

    public void getEmbeddings() {

    }

}
