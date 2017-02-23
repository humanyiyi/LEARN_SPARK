package com.udbac.ua.spark.mr;

/**
 * Created by root on 2017/2/20.
 */
public class UAinfo {
    private String catalog;
    private String browser;
    private String browserver;
    private String os;
    private String osver;
    private String device;
    private String brand;
    private String model;

    public void setCatalog(String catalog) {
        this.catalog = catalog;
    }

    public void setBrowser(String browser) {
        this.browser = browser;
    }

    public void setBrowserver(String browserver) {
        this.browserver = browserver;
    }

    public void setOs(String os) {
        this.os = os;
    }

    public void setOsver(String osver) {
        this.osver = osver;
    }

    public void setDevice(String device) {
        this.device = device;
    }

    public void setBrand(String brand) {
        this.brand = brand;
    }

    public void setModel(String model) {
        this.model = model;
    }

    @Override
    public String toString() {
        return  catalog + '\t' +
                browser + '\t' +
                browserver + '\t' +
                os + '\t' +
                osver + '\t' +
                device + '\t' +
                brand + '\t' +
                model;
    }
}
