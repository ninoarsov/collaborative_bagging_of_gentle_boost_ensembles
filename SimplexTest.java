
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.io.FileReader;
import java.io.PrintWriter;

// http://uva.onlinejudge.org/external/8/802.html
// public class LeadOrGold {
//
//     static double DELTA = 1e-9;
//     static TwoPhaseSimplex tps;
//     static boolean minimize;
//     static double[] targetCoefficients;
//     static Constraint[] constraints;
//     static double[] targetCoefficientValuesExpected;
//     static double expected;
//     static int status;
//     static double TOLERANCE = 0.000000001;
//
//     static void initTwoPhaseSimplex() {
//
//         tps = new TwoPhaseSimplex();
//
//         tps.setObjective(targetCoefficients, minimize);
//
//         double[][] constraintArray = new double[constraints.length][targetCoefficients.length];
//         int[] equations = new int[constraints.length];
//         double[] rhs = new double[constraints.length];
//
//         for (int i = 0; i < constraints.length; ++i) {
//
//             constraintArray[i] = constraints[i].getCoefficients();
//             equations[i] = constraints[i].getEquations();
//             rhs[i] = constraints[i].getRHS();
//         }
//
//         tps.setConstraints(constraintArray, equations, rhs);
//
//         tps.init();
//     }
//
//     static void solve() {
//
//         while ((status = tps.iterate()) == TwoPhaseSimplex.CONTINUE) {
//             //System.out.println(tps.toString());
//         }
//     }
//
//     public static void main(String[] args) throws Exception {
//         int i, j, k = 0;
//
//         BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
//         StringTokenizer st;
//         boolean first = true;
//
//         while (true) {
//             int N = Integer.parseInt(br.readLine());
//             if (N == 0) {
//                 break;
//             }
//
//             if (first == true) {
//                 first = false;
//             } else {
//                 System.out.println();
//             }
//
//             int a[] = new int[N + 1];
//             int b[] = new int[N + 1];
//             int c[] = new int[N + 1];
//             for (i = 0; i <= N; i++) {
//                 st = new StringTokenizer(br.readLine());
//                 a[i] = Integer.parseInt(st.nextToken());
//                 b[i] = Integer.parseInt(st.nextToken());
//                 c[i] = Integer.parseInt(st.nextToken());
//             }
//
//             // our solution goes here
//             minimize = true;
//
//             targetCoefficients = new double[N + 1];
//             targetCoefficients[N] = 1;
//
//             ArrayList<Constraint> arr = new ArrayList<Constraint>();
//             constraints = new Constraint[3 + N + 1];
//
//
//             double coefficients[] = new double[N + 1];
//             for (i = 0; i < N; i++) {
//                 coefficients[i] = a[i];
//             }
//             coefficients[N] = -a[N];
//             arr.add(new Constraint(coefficients, TwoPhaseSimplex.GREATER_THAN, 0-TOLERANCE));
//             arr.add(new Constraint(coefficients, TwoPhaseSimplex.LESS_THAN, 0+TOLERANCE));
//
//             coefficients = new double[N + 1];
//             for (i = 0; i < N; i++) {
//                 coefficients[i] = b[i];
//             }
//             coefficients[N] = -b[N];
//
//             arr.add(new Constraint(coefficients, TwoPhaseSimplex.GREATER_THAN, 0-TOLERANCE));
//             arr.add(new Constraint(coefficients, TwoPhaseSimplex.LESS_THAN, 0+TOLERANCE));
//
//             coefficients = new double[N + 1];
//             for (i = 0; i < N; i++) {
//                 coefficients[i] = c[i];
//             }
//             coefficients[N] = -c[N];
//             arr.add(new Constraint(coefficients, TwoPhaseSimplex.GREATER_THAN, 0-TOLERANCE));
//             arr.add(new Constraint(coefficients, TwoPhaseSimplex.LESS_THAN, 0+TOLERANCE));
//
//             for (i = 0; i <= N; i++) {
//                 coefficients = new double[N+1];
//                 coefficients[i] = 1;
//                 arr.add(new Constraint(coefficients, TwoPhaseSimplex.GREATER_THAN, 1));
//             }
//
//             constraints = new Constraint[arr.size()];
//             arr.toArray(constraints);
//
//             initTwoPhaseSimplex();
//             solve();
//
//             k++;
//             System.out.println("Mixture "+k);
//             if (status == TwoPhaseSimplex.OPTIMAL) {
//                 System.out.println("Possible");
//             } else {
//                 System.out.println("Impossible");
//             }
//
//         }
//
//         br.close();
//
//     }
// }

public class SimplexTest {

    static double DELTA = 1e-9;
    static TwoPhaseSimplex tps;
    static boolean minimize;
    static double[] targetCoefficients;
    static Constraint[] constraints;
    static double[] targetCoefficientValuesExpected;
    static double expected;
    static int status;
    static double TOLERANCE = 0.000000001;

    static void initTwoPhaseSimplex() {

        tps = new TwoPhaseSimplex();

        tps.setObjective(targetCoefficients, minimize);

        double[][] constraintArray = new double[constraints.length][targetCoefficients.length];
        int[] equations = new int[constraints.length];
        double[] rhs = new double[constraints.length];

        for (int i = 0; i < constraints.length; ++i) {

            constraintArray[i] = constraints[i].getCoefficients();
            equations[i] = constraints[i].getEquations();
            rhs[i] = constraints[i].getRHS();
        }

        tps.setConstraints(constraintArray, equations, rhs);

        tps.init();
    }

    static void solve() {

        while ((status = tps.iterate()) == TwoPhaseSimplex.CONTINUE) {
            //System.out.println(tps.toString());
        }
    }

    public static void main(String[] args) throws Exception {

        int instances = 0, ensembles = 1;
        String line;
        ArrayList<Double> costs = new ArrayList<Double>();
        BufferedReader br = new BufferedReader(new FileReader("simplex_input"));

        while((line = br.readLine()) != null) {
            instances++;
            String[] nums = line.trim().split("\\s+");
            ensembles = nums.length;
            for(String num : nums) {
                double numDbl = Double.parseDouble(num);
                costs.add(numDbl);
            }
        }
        br.close();

        minimize = false ;

        ArrayList<Constraint> arr = new ArrayList<Constraint>();

        targetCoefficients = new double[costs.size()];
        for(int i = 0; i < costs.size(); i++)
            targetCoefficients[i] = costs.get(i);

        int instancesFromEachEnsemble = (int) ((double)instances / (double)ensembles);

        // now we add all constraints (each instance goes to a single ensemble)
        for(int i = 0; i < instances; i++) {
            double[] coefficients = new double[targetCoefficients.length];
            for(int j = 0; j < ensembles; j++) {
                // i-th instance, j-th ensemble (one instance goes to just a single ensemble)
                coefficients[i * ensembles + j] = 1;    //[i][j]
            }
            arr.add(new Constraint(coefficients, TwoPhaseSimplex.GREATER_THAN, 1));
            arr.add(new Constraint(coefficients, TwoPhaseSimplex.LESS_THAN, 1));
            //for some i: xi1+xi2+...+xie = 1 (goes to only one ensemble)
        }

        // additional constraints (each ensemble receives exactly instances/ensembles instances)
        for(int j = 0; j < ensembles; j++) {
            double[] coefficients = new double[targetCoefficients.length];
            for(int i = 0; i < instances; i++) {
                coefficients[i * ensembles + j] = 1;
            }
            arr.add(new Constraint(coefficients, TwoPhaseSimplex.GREATER_THAN, instancesFromEachEnsemble));
            arr.add(new Constraint(coefficients, TwoPhaseSimplex.LESS_THAN, instancesFromEachEnsemble));

            // individual benefit constraints
            coefficients = new double[targetCoefficients.length + 1];
            for(int i = 0; i < instances; i++) {
                if(i != j)
                    coefficients[i * ensembles + j] = targetCoefficients[i * ensembles + j];
                else
                    coefficients[i * ensembles + j] = 0.0;
            }
            coefficients[coefficients.length-1] = -targetCoefficients[j*ensembles + j];
            arr.add(new Constraint(coefficients, TwoPhaseSimplex.GREATER_THAN, 0+TOLERANCE));
        }




        // compute default cost
        double defaultCost = 0;
        for(int j = 0; j < ensembles; j++) {
                defaultCost += targetCoefficients[j * ensembles + j];

        }
        // System.out.println("DEFAULT COST = " + defaultCost);
        // default constraints not needed

        constraints = new Constraint[arr.size()];
        arr.toArray(constraints);

        initTwoPhaseSimplex();
        solve();

        for(int j = 0; j < ensembles; j++) {
            int s = 0;
            for(int i = 0; i < instances; i++) {
                if(tps.getCoefficients()[i * ensembles + j]> 0) s++;
            }
        }

        for(int i = 0; i < instances; i++) {
            int s = 0;
            for(int j = 0; j < ensembles; j++) {
                if(tps.getCoefficients()[i * ensembles + j] > 0) s++;
            }
        }

        // System.out.println("MAXIMIZED COST = " + tps.getObjectiveResult());

        StringBuilder sb = new StringBuilder();

        for(int i = 0; i < instances; i++) {
            for(int j = 0; j < ensembles; j++) {
                double c = (tps.getCoefficients())[i * ensembles + j];
                int cRounded = (int) (c + 0.5);
                sb.append(cRounded);
                if(j < ensembles - 1) sb.append(" ");
            }
            sb.append("\n");
        }



        PrintWriter pw = new PrintWriter("simplex_result");
        pw.print(sb.toString());
        pw.close();
    }
}

class Constraint {

    private double[] coefficients;
    private int equations;
    private double rhs;

    public Constraint(double[] coefficients, int equations, double rhs) {
        this.coefficients = coefficients;
        this.equations = equations;
        this.rhs = rhs;
    }

    public double[] getCoefficients() {
        return coefficients;
    }

    public int getEquations() {
        return equations;
    }

    public double getRHS() {
        return rhs;
    }
}

abstract class AbstractSimplex {

    public final static int LESS_THAN = 0;
    public final static int GREATER_THAN = 1;
    public final static int EQUAL_TO = 2;
    public final static int CONTINUE = 0;
    public final static int OPTIMAL = 1;
    public final static int UNBOUNDED = 2;
    protected boolean minimize;
    protected double[] objective;
    protected double[][] constraints;
    protected int[] equations;
    protected double[] rhs;
    protected double[][] m;
    protected int[] basisVariable;
    protected int[] nonBasisVariable;
    protected int[] slackVariable;
    protected boolean[] locked;

    public void init() {
        this.m = new double[constraints.length + 1][objective.length + constraints.length + 1];
        for (int i = 0; i < constraints.length; ++i) {
            for (int j = 0; j < constraints[i].length; ++j) {
                m[i][j] = constraints[i][j] * (equations[i] == GREATER_THAN ? -1 : 1);
            }
            m[i][objective.length + i] = 1;
            m[i][m[i].length - 1] = rhs[i] * (equations[i] == GREATER_THAN ? -1 : 1);
        }
        for (int i = 0; i < objective.length; ++i) {
            m[m.length - 1][i] = objective[i] * (minimize ? 1 : -1);
        }
        this.nonBasisVariable = new int[objective.length + constraints.length];
        this.slackVariable = new int[constraints.length];
        for (int i = 0; i < this.nonBasisVariable.length; ++i) {
            this.nonBasisVariable[i] = i;
            if (i >= objective.length) {
                slackVariable[i - objective.length] = i;
            }
        }
        this.basisVariable = new int[constraints.length];
        for (int i = 0; i < this.basisVariable.length; ++i) {
            basisVariable[i] = slackVariable[i];
        }
        this.locked = new boolean[basisVariable.length];
    }

    public void setObjective(double[] objective, boolean minimize) {
        this.objective = objective;
        this.minimize = minimize;
    }

    public void setConstraints(double[][] constraints, int[] equations, double[] rhs) {
        this.constraints = constraints;
        this.equations = equations;
        this.rhs = rhs;
    }

    protected void pivot(int pivotRow, int pivotColumn) {
        double quotient = m[pivotRow][pivotColumn];
        for (int i = 0; i < m[pivotRow].length; ++i) {
            m[pivotRow][i] = m[pivotRow][i] / quotient;
        }
        for (int i = 0; i < m.length; ++i) {
            if (m[i][pivotColumn] != 0 && i != pivotRow) {

                quotient = m[i][pivotColumn] / m[pivotRow][pivotColumn];

                for (int j = 0; j < m[i].length; ++j) {
                    m[i][j] = m[i][j] - quotient * m[pivotRow][j];
                }
            }
        }
        basisVariable[pivotRow] = nonBasisVariable[pivotColumn];
    }

    public double getObjectiveResult() {
        return m[m.length - 1][m[m.length - 1].length - 1] * (minimize ? -1 : 1);
    }

    public double[] getCoefficients() {
        double[] result = new double[objective.length];

        for (int i = 0; i < result.length; ++i) {
            for (int j = 0; j < basisVariable.length; ++j) {
                if (i == basisVariable[j]) {
                    result[i] = m[j][m[j].length - 1];
                }
            }
        }
        return result;
    }

    public String toString() {
        StringBuffer s = new StringBuffer();

        s.append('\t');
        for (int i = 0; i < nonBasisVariable.length; ++i) {
            if (i < (nonBasisVariable.length - basisVariable.length)) {
                s.append('x');
                s.append(nonBasisVariable[i] + 1);
            } else {
                s.append('s');
                s.append(nonBasisVariable[i] - (nonBasisVariable.length - basisVariable.length) + 1);
            }
            s.append('\t');
        }
        s.append('\n');

        for (int i = 0; i < m.length - 1; ++i) {
            if (basisVariable[i] < (nonBasisVariable.length - basisVariable.length)) {
                s.append('x');
                s.append(basisVariable[i] + 1);
            } else {
                s.append('s');
                s.append(basisVariable[i] - (nonBasisVariable.length - basisVariable.length) + 1);
            }
            s.append('\t');
            for (int j = 0; j < m[i].length; ++j) {
                s.append(m[i][j]);
                s.append('\t');
            }
            s.append('\n');
        }

        s.append('Z');
        s.append('\t');
        for (int i = 0; i < m[m.length - 1].length; ++i) {
            s.append(m[m.length - 1][i]);
            s.append('\t');
        }
        s.append('\n');


        return s.toString();
    }
}

class DualSimplex extends PrimalSimplex {

    private boolean primal;

    public void init() {
        super.init();
        this.primal = false;
    }

    public int iterate() {

        if (primal) {
            return super.iterate();
        }

        double quotient;

        // Select pivot row
        int pr = -1;
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < m.length - 1; ++i) {
            if (m[i][m[i].length - 1] < 0
                    && m[i][m[i].length - 1] < min) {

                pr = i;
                min = m[i][m[i].length - 1];
            }
        }
        if (pr < 0) {
            for (int i = 0; i < m[m.length - 1].length - 1; ++i) {
                if (m[m.length - 1][i] < 0) {
                    // Start primal
                    //System.out.println("Continue with primal simplex");
                    primal = true;
                    return CONTINUE;
                }
            }
            return OPTIMAL;
        }

        // Select pivot column
        int pc = -1;
        double max = Double.NEGATIVE_INFINITY;
        if (pr > -1) {
            for (int i = 0; i < m[pr].length - 1; ++i) {
                if (m[pr][i] < 0
                        && (i < objective.length || !locked[i - objective.length])) {

                    quotient = m[m.length - 1][i] / m[pr][i];
                    if (quotient > max) {
                        /*
                         * 2012-07-09
                         *  Greathfully thanks to Andy Jose
                         *  for finding a bug. It have to be "max" instead of "min"
                         *
                         *  min = quotient;
                         */
                        max = quotient;
                        pc = i;
                    }
                }
            }
            if (pc < 0) {
                return UNBOUNDED;
            }
        }

        // Pivot
        //System.out.println("Pivo: row="+ (pr+1) +", column="+ (pc+1));
        pivot(pr, pc);

        return CONTINUE;
    }
}

class IntegerSimplex extends TwoPhaseSimplex {

    public final static int MAX_ITERATIONS = 100;
    public final static double DELTA = 1e-5;
    private boolean eliminateReal;
    private IntegerSimplex upperBound;
    private IntegerSimplex lowerBound;
    private boolean solvedUpper;
    private boolean solvedLower;
    private double objectiveResult;
    private double[] coeffizients;

    public void init() {
        super.init();
        this.eliminateReal = false;
        this.upperBound = lowerBound = null;
        this.solvedUpper = this.solvedLower = false;
        this.objectiveResult = minimize ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        this.coeffizients = null;
    }

    public int iterate() {

        int status;
        while ((status = super.iterate()) == CONTINUE) {
        }
        if (status == UNBOUNDED) {
            return UNBOUNDED;
        }

        //System.out.println("Eliminate real values");
        int ix = -1;
        double[] coeffizients = super.getCoefficients();
        for (int i = 0; i < coeffizients.length; ++i) {
            float f = (float) Math.round(coeffizients[i]);

            if (Math.abs((Math.abs(coeffizients[i]) - Math.abs(f))) > DELTA) {
                ix = i;
                //System.out.println("Non int value at: "+ (i+1) + " "+ coeffizients[i] + " "+ f);
                break;
            }
        }

        if (ix > -1) {
            //System.out.println("Create branch for index: "+ (ix+1));

            // Branch
            upperBound = new IntegerSimplex();
            lowerBound = new IntegerSimplex();

            upperBound.minimize = lowerBound.minimize = this.minimize;
            upperBound.objective = lowerBound.objective = this.objective;

            upperBound.constraints = new double[constraints.length + 1][objective.length];
            upperBound.equations = new int[constraints.length + 1];
            upperBound.rhs = new double[constraints.length + 1];

            lowerBound.constraints = new double[constraints.length + 1][objective.length];
            lowerBound.equations = new int[constraints.length + 1];
            lowerBound.rhs = new double[constraints.length + 1];

            for (int i = 0; i < constraints.length; ++i) {
                System.arraycopy(constraints[i], 0, upperBound.constraints[i], 0, constraints[i].length);
                System.arraycopy(constraints[i], 0, lowerBound.constraints[i], 0, constraints[i].length);

                upperBound.equations[i] = lowerBound.equations[i] = equations[i];
                upperBound.rhs[i] = lowerBound.rhs[i] = rhs[i];
            }

            upperBound.constraints[constraints.length][ix] = lowerBound.constraints[constraints.length][ix] = 1;

            upperBound.equations[constraints.length] = GREATER_THAN;
            lowerBound.equations[constraints.length] = LESS_THAN;

            upperBound.rhs[constraints.length] = Math.ceil(coeffizients[ix]);
            lowerBound.rhs[constraints.length] = Math.floor(coeffizients[ix]);

            upperBound.init();
            lowerBound.init();

            while ((status = upperBound.iterate()) == CONTINUE) {
            }
            if (status == OPTIMAL) {
                this.objectiveResult = upperBound.getObjectiveResult();
                this.coeffizients = upperBound.getCoefficients();
            }

            while ((status = lowerBound.iterate()) == CONTINUE) {
            }
            if (status == OPTIMAL) {
                if (this.coeffizients != null) {
                    if (this.minimize && lowerBound.getObjectiveResult() < objectiveResult) {
                        this.objectiveResult = lowerBound.getObjectiveResult();
                        this.coeffizients = lowerBound.getCoefficients();
                    } else if (!this.minimize && lowerBound.getObjectiveResult() > objectiveResult) {
                        this.objectiveResult = lowerBound.getObjectiveResult();
                        this.coeffizients = lowerBound.getCoefficients();
                    }
                } else {
                    this.objectiveResult = lowerBound.getObjectiveResult();
                    this.coeffizients = lowerBound.getCoefficients();
                }
            }

        } else {
            this.objectiveResult = super.getObjectiveResult();
            this.coeffizients = super.getCoefficients();
        }

        return OPTIMAL;
    }

    public double getObjectiveResult() {
        return objectiveResult;
    }

    public double[] getCoefficients() {
        return coeffizients;
    }

    public String toString() {
        if (!eliminateReal) {
            return super.toString();
        } else {
            if (!solvedUpper && upperBound != null) {
                return upperBound.toString();
            } else if (!solvedLower && lowerBound != null) {
                return lowerBound.toString();
            } else {
                return "--";
            }
        }

    }
}

class PrimalSimplex extends AbstractSimplex {

    public int iterate() {

        double quotient;

        // Select pivot column
        int pc = -1;
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < m[m.length - 1].length - 1; ++i) {
            if (m[m.length - 1][i] < 0
                    && m[m.length - 1][i] < min
                    && (i < objective.length || !locked[i - objective.length])) {

                pc = i;
                min = m[m.length - 1][i];
            }
        }
        if (pc < 0) {
            return OPTIMAL;
        }

        // Select pivot row
        int pr = -1;
        min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < m.length - 1; ++i) {
            if (m[i][pc] > 0) {
                quotient = m[i][m[i].length - 1] / m[i][pc];
                if (quotient < min) {
                    min = quotient;
                    pr = i;
                }
            }
        }
        if (pr < 0) {
            return UNBOUNDED;
        }

        // Pivot
        //System.out.println("Pivo: row="+ (pr+1) +", column="+ (pc+1));
        pivot(pr, pc);

        return CONTINUE;
    }
}

class TwoPhaseSimplex extends DualSimplex {

    // Phase 0: eliminate locked variable ( = rhs) from the basis
    // Phase 1: find suitable solution
    // Phase 2: Continue with normal simplex
    private int phase;

    public void init() {
        super.init();
        this.phase = 0;
    }

    public int iterate() {

        switch (phase) {
            case 0:
                // Check for locked basis variable
                // Row
                int pr = -1;
                for (int i = 0; i < basisVariable.length; ++i) {
                    if (basisVariable[i] >= objective.length) {
                        if (equations[basisVariable[i] - objective.length] == EQUAL_TO) {
                            pr = i;
                            break;
                        }
                    }
                }

                // Column
                int pc = -1;
                if (pr > -1) {
                    for (int i = 0; i < m[pr].length - 1; ++i) {
                        if (m[pr][i] != 0) {
                            pc = i;
                            //System.out.println("Lock column: "+ (pr+1));
                            locked[pr] = true;
                            break;
                        }
                    }
                }

                // pivot
                if (pc > -1) {
                    //System.out.println("Pivo: row="+ (pr+1) +", column="+ (pc+1));
                    pivot(pr, pc);
                    return CONTINUE;
                }
                phase++;
            //System.out.println("Phase 1: Find suitable solution");

            case 1:
                // Use algorithmus 3
                pr = -1;
                for (int i = 0; i < m.length - 1; ++i) {
                    if (m[i][m[i].length - 1] < 0) {
                        pr = i;
                        break;
                    }
                }

                pc = -1;
                if (pr > 0) {
                    for (int i = 0; i < m[pr].length - 1; ++i) {
                        if (m[pr][i] < 0 && (i < objective.length || !locked[i - objective.length])) {
                            pc = i;
                            break;
                        }
                    }
                }

                if (pc > -1) {
                    //System.out.println("Pivo: row="+ (pr+1) +", column="+ (pc+1));
                    pivot(pr, pc);
                    return CONTINUE;
                }

                phase++;
            //System.out.println("Phase 2: Continue with dual simplex");

            case 2:
                return super.iterate();
        }

        return UNBOUNDED;
    }
}
