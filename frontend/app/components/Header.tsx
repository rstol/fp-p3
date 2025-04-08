import { NavLink, useLocation } from 'react-router';
import {
  NavigationMenu,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from './ui/navigation-menu';
import { cn } from '~/lib/utils';

const headerLinks = [
  { title: 'Play Analyzer', to: '/' },
  { title: 'Tagged Plays', to: '/tagged_plays' },
];

export default function Header() {
  const location = useLocation();
  return (
    <NavigationMenu className="max-w-none justify-start border-b">
      <NavigationMenuList className="gap-0">
        {headerLinks.map((link) => {
          return (
            <NavLink
              key={link.to}
              to={`${link.to}${location.search}`}
              className={({ isActive }) =>
                cn(
                  navigationMenuTriggerStyle(),
                  'rounded-none px-6 py-3',
                  isActive ? 'border-primary border-b-2' : 'text-muted-foreground',
                )
              }
            >
              {link.title}
            </NavLink>
          );
        })}
      </NavigationMenuList>
    </NavigationMenu>
  );
}
