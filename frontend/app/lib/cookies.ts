import Cookies from 'js-cookie';

export function setCookie(name: string, value: string, days: number) {
  Cookies.set(name, value, { expires: days });
}

export function getCookie(name: string): string | undefined {
  return Cookies.get(name);
}