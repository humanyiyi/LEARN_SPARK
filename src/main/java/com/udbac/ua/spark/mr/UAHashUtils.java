package com.udbac.ua.spark.mr;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.lang.StringUtils;
import ua_parser.Client;
import ua_parser.Parser;

import java.io.IOException;

/**
 * Created by root on 2017/2/14.
 */
public class UAHashUtils {
    private static Parser uapaser;
    static {
        try {
            uapaser = new Parser();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    protected static String hashUA(String uaString) {
        byte[] sha1ed = DigestUtils.sha(uaString);
        String base64ed = Base64.encodeBase64String(sha1ed);
        return base64ed.replaceAll("[/+_-]","").substring(0, 20);
    }

    protected static String handleUA(String uaStr) {
        String[] vs = uaStr.split("[^A-Za-z0-9_-]");
        UAinfo uAinfo = new UAinfo();
        if (vs.length > 0) {
            if (vs[0].equals("Youku")||vs[0].equals("Tudou")){
                uAinfo.setCatalog("Youku_Other");
                uAinfo.setBrowser(uaStr);
                uAinfo.setBrowserver("");
                uAinfo.setOs("Other");
                uAinfo.setOsver("");
                uAinfo.setDevice("Other");
                uAinfo.setBrand("");
                uAinfo.setModel("");
                String[] vs1 = uaStr.split("[;]");
                if (vs1.length == 5) {
                    uAinfo.setCatalog("Youku");
                    uAinfo.setBrowser(vs1[0]);
                    uAinfo.setBrowserver(vs1[1]);
                    uAinfo.setOs(vs1[2]);
                    uAinfo.setOsver(vs1[3]);
                    uAinfo.setDevice(vs1[4]);
                    //model 1...50
                    if (vs1[4].length() < 50) uAinfo.setModel(vs[4]);
                } else if (vs1.length == 4) {
                    String[] vs2 = uaStr.split("[; /]");
                    if (vs2.length == 6) {
                        uAinfo.setCatalog("Youku");
                        uAinfo.setBrowser(vs2[0]);
                        uAinfo.setBrowserver(vs2[2]);
                        uAinfo.setOs(vs2[4]);
                        uAinfo.setOsver(vs2[5]);
                    }
                }
            } else if (vs[0].equals("QYPlayer") || vs[0].equals("Cupid")) {
                uAinfo.setCatalog("iQiyi_Other");
                uAinfo.setBrowser(uaStr);
                uAinfo.setBrowserver("");
                uAinfo.setOs("Other");
                uAinfo.setOsver("");
                uAinfo.setDevice("Other");
                uAinfo.setBrand("");
                uAinfo.setModel("");
                String[] vs1 = uaStr.split("[;/]");
                if (vs1.length == 2) {
                    uAinfo.setCatalog("iQiyi");
                    uAinfo.setBrowser(vs1[0]);
                    uAinfo.setBrowserver(vs1[1]);
                } else if (vs1.length == 3) {
                    uAinfo.setCatalog("iQiyi");
                    uAinfo.setBrowser(vs1[0]);
                    uAinfo.setBrowserver(vs1[2]);
                    uAinfo.setOs(vs1[1]);
                }
            }else {
                //使用uap转换
                Client c = uapaser.parse(uaStr);
                if ("Other".equals(c.userAgent.family)) {
                    uAinfo.setCatalog("Other");
                }else {
                    uAinfo.setCatalog("Normal");
                }
                uAinfo.setBrowser(c.userAgent.family);
                uAinfo.setBrowserver(getOrElse(c.userAgent.major)
                        + getOrElse(c.userAgent.minor,".") + getOrElse(c.userAgent.patch,"."));
                uAinfo.setOs(c.os.family);
                uAinfo.setOsver(getOrElse(c.os.major)
                        + getOrElse(c.os.minor,".") + getOrElse(c.os.patch,"."));
                uAinfo.setDevice(c.device.family);
                uAinfo.setBrand(getOrElse(c.device.brand));
                uAinfo.setModel(getOrElse(c.device.model));
            }
        }
        return uAinfo.toString();
    }

    private static String getOrElse(String string) {
        return getOrElse(string, "");
    }

    private static String getOrElse(String string,String seprator) {
        if (StringUtils.isNotBlank(string)) {
            return seprator+string;
        }else {
            return "";
        }
    }

    public enum MyCounters {
        ALLLINECOUNTER
    }
}
