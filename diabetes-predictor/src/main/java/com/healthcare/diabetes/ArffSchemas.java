package com.healthcare.diabetes;

public final class ArffSchemas {
    private ArffSchemas() {}

    public static String pimaHeader() {
        return "@relation pima_diabetes\n" +
                "@attribute pregnancies numeric\n" +
                "@attribute glucose numeric\n" +
                "@attribute bloodPressure numeric\n" +
                "@attribute skinThickness numeric\n" +
                "@attribute insulin numeric\n" +
                "@attribute bmi numeric\n" +
                "@attribute diabetesPedigree numeric\n" +
                "@attribute age numeric\n" +
                "@attribute class {neg,pos}\n" +
                "@data\n"; // no instances, used as header
    }
}

