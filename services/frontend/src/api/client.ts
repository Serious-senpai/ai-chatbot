import axios from "axios";
import type { AxiosRequestConfig, AxiosResponse } from "axios";

class Client {
  private static readonly _HTTP_URL = import.meta.env.VITE_APP_BASE_API_URL;
  private static readonly _http = axios.create({
    baseURL: Client._HTTP_URL,
  });

  public constructor() {
    console.log(`_HTTP_URL: ${Client._HTTP_URL}`);
  }

  public getUri(config?: AxiosRequestConfig): string {
    return Client._http.getUri(config);
  }

  public get<T, R = AxiosResponse<T>, D = unknown>(url: string, config?: AxiosRequestConfig<D>): Promise<R> {
    return Client._http.get<T, R, D>(url, config);
  }

  public post<T, R = AxiosResponse<T>, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig<D>): Promise<R> {
    return Client._http.post<T, R, D>(url, data, config);
  }
}

const client = new Client();

export default client;