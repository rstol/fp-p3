import type { Route } from './+types/home';
import { Dummy } from '../dummy/Dummy';

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'New React Router App' },
    { name: 'description', content: 'Welcome to React Router!' },
  ];
}

export default function Home() {
  return <Dummy />;
}
