package com.udbac.ua.spark.mr;

import java.io.IOException;
import java.util.Properties;

/**
 * Created by root on 2017/1/12.
 */

public class QueryProperties {

//    public static String[] query()  {
//        Properties prop = new Properties();
//        InputStream in = null;
//        try {
//            in = new BufferedInputStream(new FileInputStream("D:\\UDBAC\\LOG_ANALYSER\\src\\main\\resources\\application.properties"));
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        }
//        try {
//            prop.load(in);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//         String param = prop.getProperty("com.udbac.hadoop.util.query");
//        String[] params = param.split(",");
//        return params;
//    }
    private static Properties prop = new Properties();
     static{
         try {
             if (QueryProperties.class.getResourceAsStream("/application.properties") == null) {
                 throw new RuntimeException("application.properties is not found ");
             }
             prop.load(QueryProperties.class.getResourceAsStream("/application.properties"));
         } catch (IOException e) {
             e.printStackTrace();
         }
     }
    public static String[] query(){

        String[] param = prop.getProperty("com.udbac.hadoop.util.query").split(",");

        return param;
    }
//    public static void main(String[] args){
//        query();
//    }
}
